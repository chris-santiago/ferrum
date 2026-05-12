"""Phase 10f — clustering / manifold diagnostics (silhouette, PCA variance, embeddings, intercluster distance)."""
from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl

from ..deps import require_sklearn, require_umap


class ClusteringMixin:
    """Phase 10f — clustering / manifold diagnostics (silhouette, PCA variance, embeddings, intercluster distance)."""

    # --- 10f: clustering / manifold --------------------------------------

    def silhouette(self, *, k: int | None = None) -> pl.DataFrame:
        """Per-sample silhouette values, sorted within cluster descending.

        Returns one row per sample with columns ``sample_id`` (original X
        index), ``y_position`` (sequential 0..n-1 stack order — used by
        ``mark_silhouette`` to render bars in a tightly-packed Rousseeuw
        layout), ``cluster``, and ``silhouette_value``.

        ``k`` is informational; if provided, the result is filtered to
        clusters in ``range(k)``.
        """
        key = self._cache_key("silhouette", k=k)
        if key in self._cache:
            return self._cache[key]
        require_sklearn("silhouette")
        from sklearn.metrics import silhouette_samples

        X_np = self._X.to_numpy()
        if "labels_" in self._capabilities:
            labels = np.asarray(self._model.labels_)
        elif "predict" in self._capabilities:
            labels = np.asarray(self._model.predict(X_np))
        else:
            raise AttributeError(
                "ModelSource.silhouette() requires the wrapped model to "
                "expose 'labels_' or 'predict'."
            )
        sv = silhouette_samples(X_np, labels)
        clusters = sorted(set(int(c) for c in labels.tolist()))
        if k is not None:
            clusters = [c for c in clusters if c < int(k)]
        rows: list[dict] = []
        y_pos = 0
        for c in clusters:
            mask = labels == c
            idxs = np.where(mask)[0]
            vals = sv[mask]
            order = np.argsort(-vals)
            for i, val in zip(idxs[order], vals[order]):
                rows.append({
                    "sample_id": int(i),
                    "y_position": int(y_pos),
                    # cluster is rendered as a categorical color channel;
                    # serialize as Utf8 to match every other 10b/10c
                    # categorical diagnostic schema (ROC class, confusion
                    # actual/predicted) and avoid the renderer's
                    # continuous-scale fallback for low-cardinality ints.
                    "cluster": str(int(c)),
                    "silhouette_value": float(val),
                })
                y_pos += 1
        df = pl.DataFrame(rows)
        self._cache[key] = df
        return df

    def pca_variance(self, *, n_components: int | None = None) -> pl.DataFrame:
        """Explained-variance ratio per principal component plus the
        cumulative running sum. Requires the wrapped model to expose
        ``explained_variance_ratio_`` (e.g. ``sklearn.decomposition.PCA``,
        ``TruncatedSVD``).
        """
        key = self._cache_key("pca_variance", n_components=n_components)
        if key in self._cache:
            return self._cache[key]
        self._require_capability("explained_variance_ratio_", "pca_variance")
        evr = np.asarray(self._model.explained_variance_ratio_, dtype=np.float64)
        if n_components is not None:
            evr = evr[: int(n_components)]
        cum = np.cumsum(evr)
        df = pl.DataFrame({
            "component": list(range(1, len(evr) + 1)),
            "explained_variance_ratio": [float(x) for x in evr],
            "cumulative_variance_ratio": [float(x) for x in cum],
        })
        self._cache[key] = df
        return df

    def embeddings(
        self,
        *,
        method: str = "umap",
        n_components: int = 2,
        **method_kwargs: Any,
    ) -> pl.DataFrame:
        """Low-dimensional embedding of X via UMAP / t-SNE / PCA.

        Returns ``dim_0`` … ``dim_{n_components-1}`` plus a ``label`` column
        (``y`` when provided, else zeros — used to color the scatter).
        ``random_state`` is taken from the source's ``random_state``.
        """
        key = self._cache_key(
            "embeddings",
            embed_method=method,
            n_components=n_components,
            kwargs=tuple(sorted(method_kwargs.items())),
        )
        if key in self._cache:
            return self._cache[key]
        X_np = self._X.to_numpy()
        seed = self._random_state if self._random_state is not None else 0
        if method == "umap":
            umap = require_umap("embeddings")
            reducer = umap.UMAP(
                n_components=n_components, random_state=seed, **method_kwargs,
            )
            emb = reducer.fit_transform(X_np)
        elif method == "tsne":
            require_sklearn("embeddings(tsne)")
            from sklearn.manifold import TSNE

            emb = TSNE(
                n_components=n_components, random_state=seed, **method_kwargs,
            ).fit_transform(X_np)
        elif method == "pca":
            require_sklearn("embeddings(pca)")
            from sklearn.decomposition import PCA

            emb = PCA(
                n_components=n_components, random_state=seed, **method_kwargs,
            ).fit_transform(X_np)
        else:
            raise ValueError(
                f"ModelSource.embeddings(method={method!r}) — expected "
                "'umap', 'tsne', or 'pca'."
            )
        emb = np.asarray(emb, dtype=np.float64)
        if self._y is not None:
            label_arr = np.asarray(self._y.to_numpy())
        else:
            label_arr = np.zeros(emb.shape[0])
        data: dict[str, Any] = {
            f"dim_{i}": [float(v) for v in emb[:, i]]
            for i in range(emb.shape[1])
        }
        data["label"] = label_arr.tolist()
        df = pl.DataFrame(data)
        self._cache[key] = df
        return df

    def intercluster_distance(
        self,
        k: int,
        *,
        method: str = "mds",
    ) -> pl.DataFrame:
        """2D embedding of cluster centers + cluster size.

        Returns one row per cluster with ``cluster`` (Int64), ``x`` / ``y``
        (Float64, the 2D embedded coordinate), and ``size`` (Int64, sample
        count). Requires the wrapped model to expose ``cluster_centers_``.
        """
        key = self._cache_key(
            "intercluster_distance", k=int(k), embed_method=method,
        )
        if key in self._cache:
            return self._cache[key]
        require_sklearn("intercluster_distance")
        self._require_capability("cluster_centers_", "intercluster_distance")
        centers = np.asarray(self._model.cluster_centers_, dtype=np.float64)
        if centers.shape[0] < int(k):
            k = centers.shape[0]
        seed = self._random_state if self._random_state is not None else 0
        if method == "mds":
            from sklearn.manifold import MDS

            xy = MDS(
                n_components=2, random_state=seed, normalized_stress="auto",
            ).fit_transform(centers[:k])
        elif method == "tsne":
            from sklearn.manifold import TSNE

            xy = TSNE(
                n_components=2, random_state=seed,
                perplexity=max(1, min(5, int(k) - 1)),
            ).fit_transform(centers[:k])
        else:
            raise ValueError(
                f"ModelSource.intercluster_distance(method={method!r}) — "
                "expected 'mds' or 'tsne'."
            )
        if "labels_" in self._capabilities:
            labels = np.asarray(self._model.labels_)
            sizes = np.bincount(labels.astype(int), minlength=int(k))[: int(k)]
        else:
            sizes = np.ones(int(k), dtype=int)
        df = pl.DataFrame({
            # cluster routes through a categorical color scale (one
            # color per cluster id); serialize as Utf8 — same Int64 →
            # continuous-scale gotcha as silhouette.cluster.
            "cluster": [str(i) for i in range(int(k))],
            "x": [float(v) for v in xy[:, 0]],
            "y": [float(v) for v in xy[:, 1]],
            "size": [int(s) for s in sizes],
        })
        self._cache[key] = df
        return df

