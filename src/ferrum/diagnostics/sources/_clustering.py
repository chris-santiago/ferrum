"""Phase 10f — clustering / manifold diagnostics (silhouette, PCA variance, embeddings, intercluster distance)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pyarrow as pa
import polars as pl

from ferrum import _core
from ferrum._validate import validate_choice

if TYPE_CHECKING:
    from ._protocols import _SourceState as _MixinBase
else:
    _MixinBase = object


class ClusteringMixin(_MixinBase):
    """Phase 10f — clustering / manifold diagnostics (silhouette, PCA variance, embeddings, intercluster distance)."""

    # --- 10f: clustering / manifold --------------------------------------

    def silhouette(self, *, k: int | None = None) -> pl.DataFrame:
        """Per-sample silhouette values, sorted within cluster descending.

        Returns one row per sample with columns ``sample_id`` (original X
        index), ``y_position`` (sequential 0..n-1 stack order — used by
        ``mark_silhouette`` to render bars in a tightly-packed Rousseeuw
        layout), ``cluster``, and ``silhouette_value``.

        Parameters
        ----------
        k : int, optional
            Informational cluster count. When provided, the result is
            filtered to clusters in ``range(k)``.
        """
        key = self._cache_key("silhouette", k=k)
        if key in self._cache:
            return self._cache[key]

        if "labels_" in self._capabilities:
            labels = np.asarray(self._model.labels_)
        elif "predict" in self._capabilities:
            labels = np.asarray(self._model.predict(self._X))
        else:
            raise AttributeError(
                "ModelSource.silhouette() requires the wrapped model to "
                "expose 'labels_' or 'predict'."
            )
        x_arrow = self._x_record_batch()
        labels_arrow = pa.array(labels.astype(int).tolist(), type=pa.int64())
        sv_raw = _core.silhouette_samples(x_arrow, labels_arrow, "euclidean")
        sv = np.array(pa.array(sv_raw), dtype=np.float64)
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
                rows.append(
                    {
                        "sample_id": int(i),
                        "y_position": int(y_pos),
                        "cluster": str(int(c)),
                        "silhouette_value": float(val),
                    }
                )
                y_pos += 1
        df = pl.DataFrame(rows)
        self._cache[key] = df
        return df

    def pca_variance(self, *, n_components: int | None = None) -> pl.DataFrame:
        """Explained-variance ratio per principal component plus the
        cumulative running sum.

        If the wrapped model exposes ``explained_variance_ratio_`` (e.g.
        ``sklearn.decomposition.PCA``), reads it directly (backward compat).
        Otherwise computes from raw X via Rust SVD.

        Parameters
        ----------
        n_components : int, optional
            Truncate the result to the first ``n_components`` components.
            ``None`` (default) keeps all components.
        """
        key = self._cache_key("pca_variance", n_components=n_components)
        if key in self._cache:
            return self._cache[key]
        if "explained_variance_ratio_" in self._capabilities:
            evr = np.asarray(self._model.explained_variance_ratio_, dtype=np.float64)
            if n_components is not None:
                evr = evr[: int(n_components)]
            cum = np.cumsum(evr)
            df = pl.DataFrame(
                {
                    "component": list(range(1, len(evr) + 1)),
                    "explained_variance_ratio": [float(x) for x in evr],
                    "cumulative_variance_ratio": [float(x) for x in cum],
                }
            )
        else:
            x_arrow = self._x_record_batch()
            result = _core.pca_variance(x_arrow, n_components)
            df = pl.from_arrow(result)
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

        Parameters
        ----------
        method : {"umap", "tsne", "pca"}, default "umap"
            Dimensionality-reduction algorithm.
        n_components : int, default 2
            Number of embedding dimensions to emit (``dim_0`` …
            ``dim_{n_components-1}``).
        **method_kwargs
            Algorithm-specific options forwarded to the Rust kernel.
            ``"umap"`` accepts ``n_neighbors``, ``min_dist``, ``n_epochs``;
            ``"tsne"`` accepts ``perplexity``, ``learning_rate``,
            ``n_iter``; ``"pca"`` takes none.
        """
        validate_choice("ModelSource.embeddings", "method", method, ("umap", "tsne", "pca"))
        key = self._cache_key(
            "embeddings",
            embed_method=method,
            n_components=n_components,
            kwargs=tuple(sorted(method_kwargs.items())),
        )
        if key in self._cache:
            return self._cache[key]
        seed = self._random_state if self._random_state is not None else 0
        if method == "umap":
            x_arrow = self._x_record_batch()
            result = _core.umap_embedding(
                x_arrow,
                n_components,
                seed,
                method_kwargs.get("n_neighbors"),
                method_kwargs.get("min_dist"),
                method_kwargs.get("n_epochs"),
            )
            df = pl.from_arrow(result)
        elif method == "tsne":
            x_arrow = self._x_record_batch()
            result = _core.tsne_embedding(
                x_arrow,
                n_components,
                seed,
                method_kwargs.get("perplexity"),
                method_kwargs.get("learning_rate"),
                method_kwargs.get("n_iter"),
            )
            df = pl.from_arrow(result)
        else:
            # method == "pca" (validated above)
            x_arrow = self._x_record_batch()
            scores_batch = _core.pca_scores(x_arrow, n_components)
            df = pl.from_arrow(scores_batch)
        if self._y is not None:
            label_arr = np.asarray(self._y)
        elif hasattr(self._model, "labels_"):
            label_arr = np.asarray(self._model.labels_)
        else:
            label_arr = np.zeros(df.height, dtype=int)
        df = df.with_columns(pl.Series("label", label_arr.astype(str).tolist()))
        self._cache[key] = df
        return df

    def intercluster_distance(
        self,
        k: int,
        *,
        method: str = "mds",
    ) -> pl.DataFrame:
        """2D embedding of cluster centers + cluster size.

        Returns one row per cluster with ``cluster`` (Utf8 — a stringified
        ``0..k-1`` index, matching ``SCHEMA_INTERCLUSTER_DISTANCE`` and the
        emitted column), ``x`` / ``y`` (Float64, the 2D embedded coordinate),
        and ``size`` (Int64, sample count). Requires the wrapped model to
        expose ``cluster_centers_``.

        Parameters
        ----------
        k : int
            Number of clusters to embed. Clamped to the number of available
            ``cluster_centers_``.
        method : {"mds", "tsne"}, default "mds"
            Embedding algorithm for projecting cluster centers to 2D.
            ``"tsne"`` requires at least 4 clusters; use ``"mds"`` for small
            ``k``.
        """
        validate_choice("ModelSource.intercluster_distance", "method", method, ("mds", "tsne"))
        key = self._cache_key(
            "intercluster_distance",
            k=int(k),
            embed_method=method,
        )
        if key in self._cache:
            return self._cache[key]
        self._require_capability("cluster_centers_", "intercluster_distance")
        centers = np.asarray(self._model.cluster_centers_, dtype=np.float64)
        if centers.shape[0] < int(k):
            k = centers.shape[0]
        seed = self._random_state if self._random_state is not None else 0
        if method == "mds":
            centers_arrow = pa.RecordBatch.from_pydict(
                {f"f{j}": centers[:k, j].tolist() for j in range(centers.shape[1])}
            )
            mds_batch = _core.mds_classical(centers_arrow, 2, "features", "euclidean")
            mds_df = pl.from_arrow(mds_batch)
            xy = np.column_stack(
                [
                    mds_df["dim_0"].to_numpy(),
                    mds_df["dim_1"].to_numpy(),
                ]
            )
        else:
            # method == "tsne" (validated above)
            if int(k) < 4:
                raise ValueError(
                    f"intercluster_distance(method='tsne') requires at least 4 "
                    f"clusters, got k={k}. Use method='mds' for small k."
                )
            centers_arrow = pa.RecordBatch.from_pydict(
                {f"f{j}": centers[:k, j].tolist() for j in range(centers.shape[1])}
            )
            perplexity = float(max(1, min(5, int(k) - 1)))
            tsne_batch = _core.tsne_embedding(
                centers_arrow,
                2,
                seed,
                perplexity,
            )
            tsne_df = pl.from_arrow(tsne_batch)
            xy = np.column_stack(
                [
                    tsne_df["dim_0"].to_numpy(),
                    tsne_df["dim_1"].to_numpy(),
                ]
            )
        if "labels_" in self._capabilities:
            labels = np.asarray(self._model.labels_)
            sizes = np.bincount(labels.astype(int), minlength=int(k))[: int(k)]
        else:
            sizes = np.ones(int(k), dtype=int)
        df = pl.DataFrame(
            {
                "cluster": [str(i) for i in range(int(k))],
                "x": [float(v) for v in xy[:, 0]],
                "y": [float(v) for v in xy[:, 1]],
                "size": [int(s) for s in sizes],
            }
        )
        self._cache[key] = df
        return df
