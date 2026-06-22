"""Clustering domain figure functions and their private builders.

Public API
----------
pca_scree_chart, cluster_diagnostics, intercluster_distance_chart,
silhouette_chart, manifold_chart, elbow_chart.

Each public function wraps ``_resolve_source`` (shared across all figure
functions) and dispatches to a co-located ``_*_from_source`` builder that
produces a fully-formed ``Chart``.

Internal-only builders
----------------------
_silhouette_chart_from_source, _pca_scree_chart_from_source,
_pca_scree_chart_from_variance_df, _intercluster_distance_chart_from_source,
_cluster_diagnostics_chart.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import polars as pl

if TYPE_CHECKING:
    from ferrum import Chart, HConcatChart

from ferrum.encoding import X, Y
from ferrum._overrides import _apply_overrides
from ferrum.plots._helpers import _finalize_chart, _resolve_source, _validate_choice


# ---------------------------------------------------------------------------
# Builders (private)
# ---------------------------------------------------------------------------


def _silhouette_chart_from_source(
    source: Any,
    *,
    k: int | None = None,
    subtitle: str | None = None,
    mark: dict | None = None,
    encode: dict | None = None,
    properties: dict | None = None,
    layers: list | None = None,
    theme: Any = None,
):
    """Silhouette chart from a ModelSource. The source method packs
    samples into a 0..n-1 ``y_position`` stack order so the bars render
    tightly per cluster.

    ``subtitle`` is opt-in: the silhouette mark sets no chart title, so a
    subtitle is only attached (under a ``"Silhouette Plot"`` title) when a
    caller passes one. ``subtitle=None`` (default) leaves the chart
    title-less, byte-identical to the prior output.
    """
    import ferrum

    df = source.silhouette(k=k)
    chart = ferrum.Chart(df).mark_silhouette()
    if subtitle is not None:
        chart = chart.properties(
            title=ferrum.Title("Silhouette Plot", subtitle=subtitle),
        )
    return _finalize_chart(
        chart, mark=mark, encode=encode, properties=properties, layers=layers, theme=theme
    )


def _pca_scree_chart_from_source(
    source: Any,
    *,
    n_components: int | None = None,
    cumulative_line: bool = True,
    threshold: float | None = 0.95,
    subtitle: str | None = None,
    mark: dict | None = None,
    encode: dict | None = None,
    properties: dict | None = None,
    layers: list | None = None,
    theme: Any = None,
):
    """PCA scree chart with optional cumulative line + threshold rule.

    ``subtitle`` is opt-in: it threads into the existing ``"PCA Explained
    Variance"`` title; ``subtitle=None`` (default) is byte-identical to the
    prior output.
    """
    import ferrum
    from ferrum._layer import _Layer

    df = source.pca_variance(n_components=n_components)
    chart = ferrum.Chart(df).mark_pca_scree(
        cumulative_line=cumulative_line,
        threshold_line=threshold,
    )
    chart = chart.properties(
        title=ferrum.Title("PCA Explained Variance", subtitle=subtitle),
    )
    if cumulative_line:
        chart = chart.layer(
            _Layer(
                mark="point",
                encoding={"x": "component", "y": "cumulative_variance_ratio"},
                mark_kwargs={"size": 40, "filled": True},
                name="point",
            )
        )
    return _finalize_chart(
        chart, mark=mark, encode=encode, properties=properties, layers=layers, theme=theme
    )


def _pca_scree_chart_from_variance_df(
    df: Any,
    *,
    cumulative_line: bool = True,
    threshold: float | None = 0.95,
    subtitle: str | None = None,
    mark: dict | None = None,
    encode: dict | None = None,
    properties: dict | None = None,
    layers: list | None = None,
    theme: Any = None,
):
    """PCA scree chart from a pre-computed variance DataFrame (Rust SVD path).

    ``subtitle`` is opt-in: it threads into the existing ``"PCA Explained
    Variance"`` title; ``subtitle=None`` (default) is byte-identical to the
    prior output.
    """
    import ferrum
    from ferrum._layer import _Layer

    chart = ferrum.Chart(df).mark_pca_scree(
        cumulative_line=cumulative_line,
        threshold_line=threshold,
    )
    chart = chart.properties(
        title=ferrum.Title("PCA Explained Variance", subtitle=subtitle),
    )
    if cumulative_line:
        chart = chart.layer(
            _Layer(
                mark="point",
                encoding={"x": "component", "y": "cumulative_variance_ratio"},
                mark_kwargs={"size": 40, "filled": True},
                name="point",
            )
        )
    return _finalize_chart(
        chart, mark=mark, encode=encode, properties=properties, layers=layers, theme=theme
    )


def _intercluster_distance_chart_from_source(
    source: Any,
    *,
    k: int,
    method: str = "mds",
    mark: dict | None = None,
    encode: dict | None = None,
    properties: dict | None = None,
    layers: list | None = None,
    theme: Any = None,
):
    """Cluster-center 2D scatter sized by cluster count.

    Adds 15% padding around the data x/y range so the large
    size-encoded points don't clip at the axis edges (MDS pushes
    cluster centers to the extreme corners of the embedded space).
    """
    import ferrum

    df = source.intercluster_distance(k, method=method)
    x_lo = float(df["x"].min() or 0.0)
    x_hi = float(df["x"].max() or 0.0)
    y_lo = float(df["y"].min() or 0.0)
    y_hi = float(df["y"].max() or 0.0)
    x_pad = max((x_hi - x_lo) * 0.15, 0.5)
    y_pad = max((y_hi - y_lo) * 0.15, 0.5)
    chart = (
        ferrum.Chart(df)
        .mark_intercluster_distance()
        .encode(
            x=X(
                "x",
                title=f"{method.upper()}1",
                scale={
                    "type": "linear",
                    "domain": [x_lo - x_pad, x_hi + x_pad],
                },
            ),
            y=Y(
                "y",
                title=f"{method.upper()}2",
                scale={
                    "type": "linear",
                    "domain": [y_lo - y_pad, y_hi + y_pad],
                },
            ),
        )
    )
    chart = chart.properties(title=ferrum.Title("Intercluster Distance Map"))
    return _finalize_chart(
        chart, mark=mark, encode=encode, properties=properties, layers=layers, theme=theme
    )


def _cluster_diagnostics_chart(
    X: Any,  # noqa: N803 — matches public API parameter name
    *,
    ks: Any,
    method: str,
    scoring: str,
    n_init: int,
    random_state: int | None,
    mark: dict | None = None,
    encode: dict | None = None,
    properties: dict | None = None,
    layers: list | None = None,
    theme: Any = None,
):
    """Build the elbow / silhouette / both panel(s) for cluster_diagnostics.

    Unlike the other builders, this one sweeps the clusterer over the
    requested ``ks`` rather than wrapping a single fitted ``ModelSource``.
    See ``ferrum.cluster_diagnostics`` for the user-facing parameter docs.
    """
    import ferrum
    import numpy as np
    import pyarrow as pa
    from ferrum import _core
    from sklearn.cluster import KMeans, AgglomerativeClustering
    from ferrum.encoding import X as XEnc, Y as YEnc
    from ferrum._layer import _Layer

    # numpy required: manual inertia computation uses 2D positional indexing and mask ops.
    X_np = np.asarray(X, dtype=np.float64)
    X_np = np.ascontiguousarray(X_np)
    x_arrow = pa.RecordBatch.from_pydict(
        {f"f{j}": X_np[:, j].tolist() for j in range(X_np.shape[1])}
    )
    seed = 0 if random_state is None else int(random_state)
    rows: list[dict] = []
    for k in ks:
        if method == "kmeans":
            m = KMeans(
                n_clusters=int(k),
                n_init=int(n_init),
                random_state=seed,
            ).fit(X_np)
            inertia = float(m.inertia_)
            labels = m.labels_
        else:  # hierarchical
            m = AgglomerativeClustering(n_clusters=int(k), linkage="ward").fit(X_np)
            labels = m.labels_
            inertia = 0.0
            for cluster_id in np.unique(labels):
                mask = labels == cluster_id
                centroid = X_np[mask].mean(axis=0)
                inertia += float(np.sum((X_np[mask] - centroid) ** 2))
        labels_arrow = pa.array(labels.astype(int).tolist(), type=pa.int64())
        rows.append(
            {
                "k": int(k),
                "inertia": inertia,
                "silhouette": float(_core.silhouette_score(x_arrow, labels_arrow, "euclidean")),
            }
        )
    df = pl.DataFrame(rows)

    # Schwabish C11: elbow detection via second derivative of inertia.
    # Inject sentinel columns for the elbow vline + text before building
    # the chart so both layers share the same DataFrame.
    inertias = np.array([r["inertia"] for r in rows])
    ks_list = list(ks)
    elbow_layers: list[Layer] = []
    if len(inertias) >= 3:
        second_diff = np.diff(inertias, n=2)
        elbow_detect_idx = int(np.argmax(second_diff)) + 1
        elbow_k_val = int(ks_list[elbow_detect_idx])
        elbow_score_val = float(inertias[elbow_detect_idx])
        n = df.height
        df = df.with_columns(
            pl.Series("_elbow_k", [float(elbow_k_val)] + [None] * (n - 1), dtype=pl.Float64),
            pl.Series("_elbow_y", [elbow_score_val] + [None] * (n - 1), dtype=pl.Float64),
            pl.Series(
                "_elbow_text",
                [f"elbow at k={elbow_k_val}"] + [None] * (n - 1),
                dtype=pl.Utf8,
            ),
        )
        elbow_layers = [
            _Layer(
                mark="rule",
                encoding={"x": "_elbow_k"},
                mark_kwargs={"stroke": "#AAAAAA", "stroke_dash": [4, 4]},
                name="elbow_rule",
            ),
            _Layer(
                mark="text",
                encoding={"x": "_elbow_k", "y": "_elbow_y", "text": "_elbow_text"},
                mark_kwargs={"dx": 8, "dy": -8, "align": "left"},
                name="elbow_text",
            ),
        ]
    elbow = (
        ferrum.Chart(df)
        .mark_line()
        .encode(x=XEnc("k", title="k"), y=YEnc("inertia", title="Distortion score"))
        .properties(title=ferrum.Title("Distortion Score Elbow"))
        .layer(
            _Layer(
                mark="point",
                encoding={"x": "k", "y": "inertia"},
                mark_kwargs={"size": 40, "filled": True},
                name="point",
            ),
            *elbow_layers,
        )
    )
    sil = (
        ferrum.Chart(df)
        .mark_line()
        .encode(x=XEnc("k", title="k"), y=YEnc("silhouette", title="Silhouette score"))
        .properties(title=ferrum.Title("Silhouette Score"))
        .layer(
            _Layer(
                mark="point",
                encoding={"x": "k", "y": "silhouette"},
                mark_kwargs={"size": 40, "filled": True},
                name="point",
            )
        )
    )
    if scoring == "elbow":
        chart = elbow
    elif scoring == "silhouette":
        chart = sil
    else:
        # scoring="both" returns HConcatChart — mark/encode/layers overrides
        # don't apply meaningfully to compound views. Only properties= is
        # forwarded (via _apply_overrides, which calls .properties() on the
        # compound). mark=/encode=/layers= are silently ignored.
        chart = elbow | sil
    return _finalize_chart(
        chart, mark=mark, encode=encode, properties=properties, layers=layers, theme=theme
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def pca_scree_chart(
    model: Any,
    X: Any = None,  # noqa: N803 — matches ferrum-spec.md parameter name
    *,
    n_components: int | None = None,
    cumulative_line: bool = True,
    threshold: float | None = 0.95,
    subtitle: str | None = None,
    random_state: int | None = None,
    mark: dict | None = None,
    encode: dict | None = None,
    properties: dict | None = None,
    layers: list | None = None,
    theme: Any = None,
) -> "Chart":
    """PCA scree chart showing explained variance per component.

    Plots per-component explained variance ratio as bars, with an
    optional cumulative-variance overlay line and a horizontal
    reference rule at a target cumulative-variance threshold.

    Parameters
    ----------
    model : PCA estimator, ModelSource, or DataFrame
        A fitted ``sklearn.decomposition.PCA`` instance, an explicit
        ``ferrum.ModelSource`` wrapping one, an unfitted PCA
        estimator (fit is run on ``X``), or a raw polars/pandas
        DataFrame (variance computed via Rust SVD -- no sklearn needed).
    X : array-like, optional
        Feature matrix. Required when ``model`` is a raw
        (unfitted) estimator; ignored when it is already a
        ``ModelSource`` with data bound.
    n_components : int or None, default None
        Number of components to display. When ``None``, all available
        components from the fitted PCA are shown.
    cumulative_line : bool, default True
        When ``True``, overlays a cumulative explained-variance line.
    threshold : float or None, default 0.95
        When a float is given, draws a horizontal reference rule at
        that cumulative-variance level (e.g. ``0.95`` marks where 95%
        of variance is explained). Pass ``None`` to omit the rule.
    subtitle : str or None, default None
        Optional subtitle rendered beneath the active chart title.
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
        PCA scree bar chart with optional cumulative line and threshold
        rule.

    Examples
    --------
    >>> import ferrum as fm
    >>> from sklearn.decomposition import PCA
    >>> fm.pca_scree_chart(PCA(n_components=10).fit(X_train), threshold=0.90)

    Raw DataFrame (no sklearn required):

    >>> fm.pca_scree_chart(X_train, n_components=10)
    """
    import numpy as np
    import pyarrow as pa

    is_raw_data = False
    if isinstance(model, pl.DataFrame):
        is_raw_data = True
    elif isinstance(model, np.ndarray) and model.ndim == 2:
        is_raw_data = True
    else:
        try:
            import pandas as pd

            if isinstance(model, pd.DataFrame):
                is_raw_data = True
        except ImportError:
            pass

    if is_raw_data:
        from ferrum import _core

        if isinstance(model, pl.DataFrame):
            x_df = model
        elif isinstance(model, np.ndarray):
            x_df = pl.from_numpy(model, schema=[f"f{i}" for i in range(model.shape[1])])
        else:
            x_df = pl.from_pandas(model)

        x_arrow = pa.RecordBatch.from_pydict({c: x_df[c].to_arrow() for c in x_df.columns})
        var_batch = _core.pca_variance(x_arrow, n_components)
        df = pl.from_arrow(var_batch)
        return _pca_scree_chart_from_variance_df(
            df,
            cumulative_line=cumulative_line,
            threshold=threshold,
            subtitle=subtitle,
            mark=mark,
            encode=encode,
            properties=properties,
            layers=layers,
            theme=theme,
        )

    source = _resolve_source(model, X, None, random_state=random_state)
    return _pca_scree_chart_from_source(
        source,
        n_components=n_components,
        cumulative_line=cumulative_line,
        threshold=threshold,
        subtitle=subtitle,
        mark=mark,
        encode=encode,
        properties=properties,
        layers=layers,
        theme=theme,
    )


def cluster_diagnostics(
    X: Any,  # noqa: N803 — matches ferrum-spec.md parameter name
    *,
    ks: Any,
    method: str = "kmeans",
    scoring: str = "both",
    n_init: int = 10,
    random_state: int | None = None,
    mark: dict | None = None,
    encode: dict | None = None,
    properties: dict | None = None,
    layers: list | None = None,
    theme: Any = None,
) -> "Chart | HConcatChart":
    """Elbow and silhouette diagnostics over a range of cluster counts.

    Fits one clusterer per value of k and renders the requested
    diagnostic panel(s): distortion (inertia) vs. k, mean silhouette
    score vs. k, or both side-by-side. Unlike the other figure
    functions, this sweeps the model class itself rather than wrapping
    a single pre-fitted ``ModelSource``.

    Parameters
    ----------
    X : array-like
        Feature matrix. All samples are used for fitting and scoring.
        Polars DataFrames, pandas DataFrames, and 2D numpy arrays are
        accepted.
    ks : iterable of int
        Values of k (number of clusters) to evaluate.
    method : {"kmeans", "hierarchical"}, default "kmeans"
        Clustering algorithm.

        - ``"kmeans"`` -- ``sklearn.cluster.KMeans``. Uses the estimator's
          native ``inertia_`` attribute.
        - ``"hierarchical"`` -- ``sklearn.cluster.AgglomerativeClustering``
          (Ward linkage). Inertia is computed manually as the sum of
          squared distances from each sample to its cluster centroid,
          since AgglomerativeClustering does not expose ``inertia_``.

        DBSCAN is intentionally not supported -- its number of clusters
        is determined by ``eps`` / ``min_samples``, not by a swept k,
        so the chart's elbow / silhouette-vs-k axis doesn't apply.
    scoring : {"elbow", "silhouette", "both"}, default "both"
        Which diagnostic panel(s) to render.

        - ``"elbow"`` -- inertia-vs-k line chart only.
        - ``"silhouette"`` -- mean silhouette-vs-k line chart only.
        - ``"both"`` -- side-by-side HConcatChart of the two.
    n_init : int, default 10
        Number of KMeans initializations per k; forwarded to
        ``sklearn.cluster.KMeans(n_init=...)``. Ignored for hierarchical
        clustering, which is deterministic given the linkage strategy.
    random_state : int or None, default None
        Random seed for KMeans initialisation. When ``None``, defaults
        to seed 0 for deterministic results. Ignored for hierarchical
        clustering.
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
    Chart or HConcatChart
        ``Chart`` when ``scoring`` is ``"elbow"`` or ``"silhouette"``;
        ``HConcatChart`` with both panels side-by-side when
        ``scoring="both"``.

    Raises
    ------
    ValueError
        If ``method`` is unknown or ``scoring`` is not in the supported
        set.

    Examples
    --------
    >>> import ferrum as fm
    >>> fm.cluster_diagnostics(X_train, ks=range(2, 11))
    >>> fm.cluster_diagnostics(X_train, ks=range(2, 11),
    ...                         method="hierarchical", scoring="silhouette")
    """
    from ferrum._diagnostics.deps import require_sklearn

    require_sklearn("cluster_diagnostics")
    _validate_choice("cluster_diagnostics", "method", method, {"kmeans", "hierarchical"})
    _validate_choice("cluster_diagnostics", "scoring", scoring, {"elbow", "silhouette", "both"})
    return _cluster_diagnostics_chart(
        X,
        ks=ks,
        method=method,
        scoring=scoring,
        n_init=n_init,
        random_state=random_state,
        mark=mark,
        encode=encode,
        properties=properties,
        layers=layers,
        theme=theme,
    )


def intercluster_distance_chart(
    model: Any,
    X: Any = None,  # noqa: N803 — matches ferrum-spec.md parameter name
    *,
    k: int | None = None,
    method: str = "mds",
    random_state: int | None = None,
    mark: dict | None = None,
    encode: dict | None = None,
    properties: dict | None = None,
    layers: list | None = None,
    theme: Any = None,
) -> "Chart":
    """Intercluster distance map: 2D embedding of cluster centers.

    Embeds cluster centers into 2D using MDS so their pairwise distances
    are approximately preserved. Each center is rendered as a
    size-encoded circle (bubble area proportional to cluster count).
    A 15% padding is added around the data range so large bubbles do
    not clip at axis edges.

    Parameters
    ----------
    model : fitted clusterer or ModelSource
        A fitted sklearn-compatible clusterer (must expose
        ``cluster_centers_`` or ``n_clusters``) or an explicit
        ``ferrum.ModelSource``.
    X : array-like, optional
        Feature matrix used to compute cluster member counts.
        Required when ``model`` is a raw estimator.
    k : int or None, default None
        Number of clusters. When ``None``, inferred from
        ``model.n_clusters`` or ``len(model.cluster_centers_)``; raises
        ``ValueError`` if neither attribute is present.
    method : {"mds", "tsne"}, default "mds"
        Dimensionality-reduction method for embedding cluster centers.
        ``"mds"`` (default) uses sklearn MDS to preserve pairwise
        distances; ``"tsne"`` uses t-SNE (perplexity clamped to
        ``min(5, k-1)`` for small cluster counts).
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
        2D scatter chart of embedded cluster centers sized by cluster
        population.

    Raises
    ------
    ValueError
        If ``k`` is ``None`` and the model exposes neither
        ``n_clusters`` nor ``cluster_centers_``.

    Examples
    --------
    >>> import ferrum as fm
    >>> from sklearn.cluster import KMeans
    >>> fm.intercluster_distance_chart(KMeans(n_clusters=5).fit(X_train), X_train)
    """
    source = _resolve_source(model, X, None, random_state=random_state)
    if k is None:
        if hasattr(source.model, "n_clusters"):
            k = int(source.model.n_clusters)
        elif hasattr(source.model, "cluster_centers_"):
            k = int(source.model.cluster_centers_.shape[0])
        else:
            raise ValueError(
                "intercluster_distance_chart(k=...) is required when the "
                "wrapped model exposes neither n_clusters nor "
                "cluster_centers_."
            )
    return _intercluster_distance_chart_from_source(
        source,
        k=int(k),
        method=method,
        mark=mark,
        encode=encode,
        properties=properties,
        layers=layers,
        theme=theme,
    )


def silhouette_chart(
    model: Any,
    X: Any = None,
    *,
    subtitle: str | None = None,
    random_state: int | None = None,
    mark: dict | None = None,
    encode: dict | None = None,
    properties: dict | None = None,
    layers: list | None = None,
    theme: Any = None,
) -> "Chart":
    """Rousseeuw silhouette plot for a fitted clusterer.

    Computes per-sample silhouette coefficients and renders a horizontal
    bar chart sorted by cluster, one bar per sample, colored by cluster
    label.

    Parameters
    ----------
    model : fitted clusterer or ModelSource
        A fitted sklearn-compatible clustering estimator that exposes
        ``labels_``, or an explicit ``ferrum.ModelSource``.
    X : array-like, optional
        Feature matrix. Required when ``model`` is a raw
        estimator; ignored when it is already a ``ModelSource``.
    subtitle : str or None, default None
        Optional subtitle. The silhouette mark sets no chart title, so a
        subtitle is only drawn (under a ``"Silhouette Plot"`` title) when
        one is supplied; the default leaves the chart title-less.
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
        Horizontal silhouette bar chart with per-cluster color encoding.

    Examples
    --------
    >>> import ferrum as fm
    >>> from sklearn.cluster import KMeans
    >>> fm.silhouette_chart(KMeans(n_clusters=3, random_state=0).fit(X), X)
    """
    source = _resolve_source(model, X, None, random_state=random_state)
    return _silhouette_chart_from_source(
        source,
        subtitle=subtitle,
        mark=mark,
        encode=encode,
        properties=properties,
        layers=layers,
        theme=theme,
    )


def manifold_chart(
    model: Any,
    X: Any = None,
    *,
    method: str = "umap",
    random_state: int | None = None,
    mark: dict | None = None,
    encode: dict | None = None,
    properties: dict | None = None,
    layers: list | None = None,
    theme: Any = None,
) -> "Chart":
    """Low-dimensional manifold-embedding scatter (UMAP / t-SNE / PCA).

    Projects the input data to two dimensions via the selected embedding
    algorithm and renders a point chart with axes ``dim_0`` / ``dim_1``
    colored by cluster label.

    Parameters
    ----------
    model : fitted clusterer or ModelSource
        A fitted clustering estimator whose ``labels_`` attribute colors
        the points, or an explicit ``ferrum.ModelSource``.
    X : array-like, optional
        Feature matrix. Required when ``model`` is a raw
        estimator; ignored when it is already a ``ModelSource``.
    method : str, default "umap"
        Embedding algorithm. Typical values are ``"umap"``, ``"tsne"``,
        and ``"pca"``.
    random_state : int or None, default None
        Seed forwarded to ``ModelSource`` for stochastic embeddings.
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
        2-D scatter plot of the embedded data, colored by cluster label.

    Examples
    --------
    >>> import ferrum as fm
    >>> from sklearn.cluster import KMeans
    >>> fm.manifold_chart(KMeans(n_clusters=4, random_state=0).fit(X), X, method="tsne")
    """
    import ferrum

    source = _resolve_source(model, X, None, random_state=random_state)
    emb = source.embeddings(method=method)
    chart = ferrum.Chart(emb).mark_point().encode(x="dim_0", y="dim_1", color="label:N")
    return _finalize_chart(
        chart, mark=mark, encode=encode, properties=properties, layers=layers, theme=theme
    )


def elbow_chart(
    model: Any,
    X: Any,
    *,
    ks: Any,
    metric: str = "distortion",
    random_state: int | None = None,
    mark: dict | None = None,
    encode: dict | None = None,
    properties: dict | None = None,
    layers: list | None = None,
    theme: Any = None,
) -> "Chart":
    """Elbow / score sweep over a range of k for a clustering algorithm.

    Fits one model per k value and plots the selected score metric against
    k. Useful for identifying the optimal number of clusters visually.

    Parameters
    ----------
    model : type
        Uninstantiated clustering class (e.g. ``sklearn.cluster.KMeans``).
        Must accept ``n_clusters``, ``random_state``, and ``n_init`` keyword
        arguments.
    X : array-like
        Feature matrix used to fit each per-k model.
    ks : sequence of int
        The candidate k values to sweep (e.g. ``range(2, 11)``).
    metric : {"distortion", "silhouette", "calinski_harabasz"}, default "distortion"
        Score to optimize. ``"distortion"`` is minimized; ``"silhouette"``
        and ``"calinski_harabasz"`` are maximized.
    random_state : int or None, default None
        Seed passed to every per-k model instantiation.
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
        Score-vs-k line chart with the optimal k annotated.

    Examples
    --------
    >>> import ferrum as fm
    >>> from sklearn.cluster import KMeans
    >>> fm.elbow_chart(KMeans, X_train, ks=range(2, 9))
    """
    from ferrum._diagnostics.visualizers.clustering import ElbowVisualizer

    viz = ElbowVisualizer(
        model,
        ks=ks,
        metric=metric,
        random_state=random_state,
        theme=theme,
    ).fit(X)

    chart = viz._chart
    return _finalize_chart(
        chart, mark=mark, encode=encode, properties=properties, layers=layers, theme=None
    )
