"""10f clustering / manifold / dimensionality visualizers.

Mirrors yellowbrick's KElbow, Silhouette, InterclusterDistance, Manifold,
and PCA-decomposition surfaces. Each visualizer wraps the appropriate
``ModelSource`` method and emits a ferrum chart via ``_build_chart``.

``ElbowVisualizer`` is the lone exception — it takes a model *class*
rather than a fitted instance and fits one clusterer per k inside its
``fit()`` method, paralleling yellowbrick's API. Every other visualizer
follows the standard ``FerrumVisualizer.fit(X[, y])`` protocol.
"""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import polars as pl

from .base import FerrumVisualizer


class SilhouetteVisualizer(FerrumVisualizer):
    """Rousseeuw silhouette plot for a fitted clusterer.

    Computes per-sample silhouette coefficients via ``ModelSource.silhouette``
    and renders a horizontal bar chart sorted by cluster, one bar per sample.
    Records ``mean_silhouette`` — the grand mean silhouette coefficient
    averaged across all samples — as the headline metric.

    Takes a *fitted* estimator (not a class); for the k-sweep variant use
    ``ElbowVisualizer``.

    Parameters
    ----------
    model : Any
        Fitted clustering estimator (e.g. ``sklearn.cluster.KMeans`` instance)
        that exposes ``labels_`` (and optionally ``cluster_centers_``).
    random_state : int, optional
        Seed forwarded to the underlying ``ModelSource``. Ignored when the
        silhouette computation is deterministic.
    theme : Theme, optional
        Per-chart theme override. Falls back to the global default when
        ``None``.

    Examples
    --------
    >>> import ferrum as fm
    >>> from sklearn.cluster import KMeans
    >>> model = KMeans(n_clusters=3, random_state=0).fit(X)
    >>> viz = fm.SilhouetteVisualizer(model).fit(X)
    >>> viz.show()
    >>> viz._metrics["mean_silhouette"]
    """

    def __init__(
        self,
        model: Any,
        *,
        random_state: int | None = None,
        theme: Any = None,
    ):
        super().__init__(model, random_state=random_state, theme=theme)

    def _materialize(self) -> None:
        sil = self._source.silhouette()
        self._metrics["mean_silhouette"] = float(
            sil["silhouette_value"].mean()
        )

    def _build_chart(self) -> Any:
        from ..charts import _silhouette_chart_from_source
        return _silhouette_chart_from_source(
            self._source, theme=self.theme,
        )


class ElbowVisualizer(FerrumVisualizer):
    """Elbow / score sweep over a range of k for a clusterer class.

    Unlike other ferrum visualizers, ``ElbowVisualizer`` takes a model *class*
    (e.g. ``KMeans``) — not a fitted instance — and constructs and fits one
    model per k value inside its own ``fit()`` override. The ``ModelSource``
    round-trip is skipped entirely; per-k models are transient and discarded
    after their score is recorded. Renders a score-vs-k line chart.

    Records ``best_k`` — the integer k whose score is optimal for the
    selected metric — as the headline metric. For ``"distortion"`` the
    optimal score is the minimum; for ``"silhouette"`` and
    ``"calinski_harabasz"`` it is the maximum (higher is better).

    Parameters
    ----------
    model_class : type
        Uninstantiated clustering class (e.g. ``sklearn.cluster.KMeans``).
        Must accept ``n_clusters``, ``random_state``, and ``n_init`` keyword
        arguments and expose ``.inertia_`` after fitting (for the
        ``"distortion"`` metric) or ``.labels_`` (for ``"silhouette"`` /
        ``"calinski_harabasz"``).
    ks : sequence of int
        The candidate k values to sweep (e.g. ``range(2, 11)``). Note that
        ``"silhouette"`` and ``"calinski_harabasz"`` are undefined at
        ``k == 1`` — such entries are silently skipped from the score sweep.
    metric : {"distortion", "silhouette", "calinski_harabasz"}, default "distortion"
        Score to optimize. ``"distortion"`` (sum of squared distances to the
        nearest centroid, i.e. ``.inertia_``) is minimized;
        ``"silhouette"`` (mean Rousseeuw silhouette coefficient) and
        ``"calinski_harabasz"`` (Calinski–Harabasz / Variance Ratio Criterion)
        are maximized. Any other value raises ``ValueError``.
    random_state : int, optional
        Integer seed passed as ``random_state`` to every per-k model
        instantiation. When ``None``, seed ``0`` is used.
    theme : Theme, optional
        Per-chart theme override. Falls back to the global default when
        ``None``.

    Examples
    --------
    >>> import ferrum as fm
    >>> from sklearn.cluster import KMeans
    >>> viz = fm.ElbowVisualizer(KMeans, ks=range(2, 9)).fit(X)
    >>> viz.show()
    >>> viz._metrics["best_k"]
    """

    _VALID_METRICS = ("distortion", "silhouette", "calinski_harabasz")

    def __init__(
        self,
        model_class: Any,
        *,
        ks: Sequence[int],
        metric: str = "distortion",
        random_state: int | None = None,
        theme: Any = None,
    ):
        super().__init__(model=None, random_state=random_state, theme=theme)
        if metric not in self._VALID_METRICS:
            raise ValueError(
                f"ElbowVisualizer(metric={metric!r}) is not valid; expected "
                f"one of {self._VALID_METRICS}."
            )
        self.model_class = model_class
        self.ks = list(ks)
        self.metric = metric

    def fit(self, X: Any, y: Any = None) -> "ElbowVisualizer":
        del y
        from ..deps import require_sklearn
        require_sklearn("ElbowVisualizer")
        import ferrum

        X_np = X.to_numpy() if hasattr(X, "to_numpy") else np.asarray(X)
        seed = 0 if self.random_state is None else int(self.random_state)
        rows: list[dict] = []
        for k in self.ks:
            k_int = int(k)
            # silhouette and calinski_harabasz are undefined at k=1.
            if self.metric in ("silhouette", "calinski_harabasz") and k_int < 2:
                continue
            m = self.model_class(
                n_clusters=k_int, random_state=seed, n_init=10,
            ).fit(X_np)
            if self.metric == "distortion":
                score = float(m.inertia_)
            elif self.metric == "silhouette":
                from sklearn.metrics import silhouette_score
                labels = getattr(m, "labels_", None)
                if labels is None:
                    labels = m.predict(X_np)
                score = float(silhouette_score(X_np, labels))
            else:  # "calinski_harabasz"
                from sklearn.metrics import calinski_harabasz_score
                labels = getattr(m, "labels_", None)
                if labels is None:
                    labels = m.predict(X_np)
                score = float(calinski_harabasz_score(X_np, labels))
            rows.append({"k": k_int, "score": score})
        if not rows:
            raise ValueError(
                f"ElbowVisualizer.fit produced no scores for metric="
                f"{self.metric!r} over ks={self.ks!r}. For "
                "silhouette/calinski_harabasz, ks must contain values >= 2."
            )
        df = pl.DataFrame(rows)
        scores = df["score"].to_numpy()
        # distortion is minimized; silhouette and CH are maximized.
        if self.metric == "distortion":
            idx = int(np.argmin(scores))
        else:
            idx = int(np.argmax(scores))
        self._metrics["best_k"] = float(df["k"][idx])
        chart = ferrum.Chart(df).mark_line().encode(x="k", y="score")
        if self.theme is not None:
            chart = chart.theme(self.theme)
        self._chart = chart
        self._fitted = True
        return self


class ManifoldVisualizer(FerrumVisualizer):
    """Low-dimensional manifold-embedding scatter (UMAP / t-SNE / PCA).

    Projects the input data to two dimensions via the method selected by
    ``method`` and renders a point chart with axes ``dim_0`` / ``dim_1``
    colored by cluster label. The embedding is computed by
    ``ModelSource.embeddings`` and cached so ``_build_chart`` does not
    recompute it.

    Takes a *fitted* clustering estimator whose ``labels_`` attribute is used
    to color points. Pass ``model=None`` only if you override ``fit`` in a
    subclass and supply ``labels_`` by other means.

    Records ``n_samples`` — the number of rows in the embedding — as the
    headline metric.

    Parameters
    ----------
    model : Any, optional
        Fitted clustering estimator (e.g. ``KMeans`` instance) that exposes
        ``labels_``. Defaults to ``None`` (for subclass overrides).
    method : str, default "umap"
        Embedding algorithm forwarded to ``ModelSource.embeddings``. Typical
        values are ``"umap"``, ``"tsne"``, and ``"pca"``.
    random_state : int, optional
        Seed forwarded to the underlying ``ModelSource``. Controls
        reproducibility for stochastic embeddings such as UMAP and t-SNE.
    theme : Theme, optional
        Per-chart theme override. Falls back to the global default when
        ``None``.

    Examples
    --------
    >>> import ferrum as fm
    >>> from sklearn.cluster import KMeans
    >>> model = KMeans(n_clusters=4, random_state=0).fit(X)
    >>> viz = fm.ManifoldVisualizer(model, method="umap").fit(X)
    >>> viz.show()
    >>> viz._metrics["n_samples"]
    """

    def __init__(
        self,
        model: Any = None,
        *,
        method: str = "umap",
        random_state: int | None = None,
        theme: Any = None,
    ):
        super().__init__(model, random_state=random_state, theme=theme)
        self.method = method

    def _materialize(self) -> None:
        # ModelSource.embeddings() memoizes via BaseSource._cache, so the
        # second call from _build_chart is a cache hit (no recomputation).
        emb = self._source.embeddings(method=self.method)
        self._metrics["n_samples"] = float(emb.height)

    def _build_chart(self) -> Any:
        import ferrum

        emb = self._source.embeddings(method=self.method)
        chart = ferrum.Chart(emb).mark_point().encode(
            x="dim_0", y="dim_1", color="label",
        )
        if self.theme is not None:
            chart = chart.theme(self.theme)
        return chart


class InterclusterDistanceVisualizer(FerrumVisualizer):
    """Cluster-center 2D embedding with cluster-size bubble overlay.

    Projects cluster centers into 2D via the algorithm selected by ``method``
    and renders a bubble chart where each bubble represents one cluster; bubble
    area encodes the cluster's sample count. Built on
    ``ferrum.intercluster_distance_chart``.

    Takes a *fitted* clustering estimator that exposes either ``n_clusters`` or
    ``cluster_centers_`` so the number of clusters can be inferred.

    Records ``max_intercluster_dist`` — the largest Euclidean distance from any
    cluster center to the centroid of all centers in the 2D embedding — as a
    rough measure of how spread-out the clusters are.

    Parameters
    ----------
    model : Any
        Fitted clustering estimator (e.g. ``KMeans`` instance) that exposes
        ``n_clusters`` or ``cluster_centers_``.
    method : str, default "mds"
        Dimensionality-reduction algorithm forwarded to
        ``ModelSource.intercluster_distance``. Typical values include
        ``"mds"`` and ``"tsne"``.
    random_state : int, optional
        Seed forwarded to the underlying ``ModelSource``. Controls
        reproducibility for stochastic layout algorithms.
    theme : Theme, optional
        Per-chart theme override. Falls back to the global default when
        ``None``.

    Examples
    --------
    >>> import ferrum as fm
    >>> from sklearn.cluster import KMeans
    >>> model = KMeans(n_clusters=5, random_state=0).fit(X)
    >>> viz = fm.InterclusterDistanceVisualizer(model).fit(X)
    >>> viz.show()
    >>> viz._metrics["max_intercluster_dist"]
    """

    def __init__(
        self,
        model: Any,
        *,
        method: str = "mds",
        random_state: int | None = None,
        theme: Any = None,
    ):
        super().__init__(model, random_state=random_state, theme=theme)
        self.method = method

    def _materialize(self) -> None:
        k = self._infer_k()
        icd = self._source.intercluster_distance(k=k, method=self.method)
        xs = icd["x"].to_numpy()
        ys = icd["y"].to_numpy()
        cx, cy = float(xs.mean()), float(ys.mean())
        self._metrics["max_intercluster_dist"] = float(
            np.max(np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2))
        )

    def _infer_k(self) -> int:
        if hasattr(self.model, "n_clusters"):
            return int(self.model.n_clusters)
        if hasattr(self.model, "cluster_centers_"):
            return int(self.model.cluster_centers_.shape[0])
        raise AttributeError(
            "InterclusterDistanceVisualizer requires the wrapped model "
            "to expose n_clusters or cluster_centers_."
        )

    def _build_chart(self) -> Any:
        import ferrum

        return ferrum.intercluster_distance_chart(
            self._source,
            k=self._infer_k(),
            method=self.method,
            theme=self.theme,
        )


class PCAVarianceVisualizer(FerrumVisualizer):
    """PCA scree plot showing explained variance per principal component.

    Retrieves per-component explained-variance ratios via
    ``ModelSource.pca_variance`` and renders a bar chart of
    explained-variance ratio vs. component index, optionally limited to the
    first ``n_components`` components. Built on ``ferrum.pca_scree_chart``.

    Takes a *fitted* decomposition estimator (e.g. ``sklearn.decomposition.PCA``
    instance) that exposes ``explained_variance_ratio_``.

    Records ``first_component_var`` — the fraction of total variance captured
    by the first principal component — as the headline metric.

    Parameters
    ----------
    model : Any
        Fitted decomposition estimator (e.g. ``PCA`` instance) that exposes
        ``explained_variance_ratio_``.
    n_components : int, optional
        Number of components to display in the scree plot. When ``None``,
        all components present in the model are shown.
    random_state : int, optional
        Seed forwarded to the underlying ``ModelSource``. Ignored when
        the PCA variance computation is deterministic.
    theme : Theme, optional
        Per-chart theme override. Falls back to the global default when
        ``None``.

    Examples
    --------
    >>> import ferrum as fm
    >>> from sklearn.decomposition import PCA
    >>> model = PCA(n_components=10).fit(X)
    >>> viz = fm.PCAVarianceVisualizer(model, n_components=5).fit(X)
    >>> viz.show()
    >>> viz._metrics["first_component_var"]
    """

    def __init__(
        self,
        model: Any,
        *,
        n_components: int | None = None,
        random_state: int | None = None,
        theme: Any = None,
    ):
        super().__init__(model, random_state=random_state, theme=theme)
        self.n_components = n_components

    def _materialize(self) -> None:
        pca = self._source.pca_variance(n_components=self.n_components)
        self._metrics["first_component_var"] = float(
            pca["explained_variance_ratio"][0]
        )

    def _build_chart(self) -> Any:
        import ferrum

        return ferrum.pca_scree_chart(
            self._source,
            n_components=self.n_components,
            theme=self.theme,
        )
