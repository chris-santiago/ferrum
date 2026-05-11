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

    Records ``mean_silhouette`` (the per-cluster mean averaged over all
    samples) as the headline metric.
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
    """Elbow / distortion sweep over a clusterer class.

    Unlike most visualizers, ``ElbowVisualizer`` takes a model *class*
    (e.g. ``KMeans``) and fits one instance per k inside ``fit()``.
    Records ``best_k`` — the k minimizing distortion (inertia) — and
    skips the ``ModelSource`` round-trip since per-k models are
    transient. Renders an inertia-vs-k line plot.
    """

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
            m = self.model_class(
                n_clusters=int(k), random_state=seed, n_init=10,
            ).fit(X_np)
            if self.metric == "distortion":
                score = float(m.inertia_)
            else:
                raise NotImplementedError(
                    f"ElbowVisualizer(metric={self.metric!r}) is not yet "
                    "implemented. Use metric='distortion' for now."
                )
            rows.append({"k": int(k), "score": score})
        df = pl.DataFrame(rows)
        scores = df["score"].to_numpy()
        idx = int(np.argmin(scores))
        self._metrics["best_k"] = float(df["k"][idx])
        chart = ferrum.Chart(df).mark_line().encode(x="k", y="score")
        if self.theme is not None:
            chart = chart.theme(self.theme)
        self._chart = chart
        self._fitted = True
        return self


class ManifoldVisualizer(FerrumVisualizer):
    """Low-dimensional embedding scatter (UMAP / t-SNE / PCA).

    Records ``n_samples`` so the repr surfaces the embedding's row count
    without forcing the chart to compute the embedding twice.
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
        emb = self._source.embeddings(method=self.method)
        self._metrics["n_samples"] = float(emb.height)
        self._cached_embedding = emb

    def _build_chart(self) -> Any:
        import ferrum

        emb = getattr(self, "_cached_embedding", None)
        if emb is None:
            emb = self._source.embeddings(method=self.method)
        chart = ferrum.Chart(emb).mark_point().encode(
            x="dim_0", y="dim_1", color="label",
        )
        if self.theme is not None:
            chart = chart.theme(self.theme)
        return chart


class InterclusterDistanceVisualizer(FerrumVisualizer):
    """Cluster-center 2D MDS embedding + cluster-size bubbles.

    Records ``max_intercluster_dist`` — the largest Euclidean distance
    from any cluster center to the centroid of all centers — as a rough
    measure of how spread the clusters are in the 2D embedding.
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
    """PCA scree plot for a fitted decomposition model.

    Records ``first_component_var`` (the fraction of total variance the
    first principal component captures) as the headline metric.
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
