"""Phase 10g — feature ranking (rank1d, rank2d)."""
from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl



class RankingMixin:
    """Phase 10g — feature ranking (rank1d, rank2d)."""

    # ---- Phase 10g: feature ranking ----

    def rank1d(self, *, algorithm: str = "shapiro") -> pl.DataFrame:
        """Univariate feature ranking.

        ``algorithm`` in ``{"shapiro", "variance", "covariance"}``. The
        Shapiro-Wilk and variance algorithms operate on X alone;
        ``"covariance"`` ranks features by absolute sample covariance with
        ``y`` and therefore requires ``y`` to be present.

        Output schema (``SCHEMA_RANK1D``): ``feature: Utf8``,
        ``score: Float64``, ``rank: Int64``. Rows are pre-sorted by descending
        score so ``rank=1`` is always the top feature.
        """
        from ..stats import (
            covariance_rank,
            rank1d_compute,
        )

        key = self._cache_key("rank1d", algorithm=algorithm)
        if key in self._cache:
            return self._cache[key]
        if algorithm == "covariance":
            if self._y is None:
                raise ValueError(
                    "ModelSource.rank1d(algorithm='covariance') requires y."
                )
            X_np = np.asarray(self._X.to_numpy(), dtype=np.float64)
            y_np = np.asarray(self._y.to_numpy(), dtype=np.float64)
            scores = covariance_rank(X_np, y_np)
            order = np.argsort(-scores, kind="mergesort")
            df = pl.DataFrame({
                "feature": [str(self._feature_names[int(i)]) for i in order],
                "score": [float(scores[int(i)]) for i in order],
                "rank": list(range(1, len(order) + 1)),
            })
        else:
            df = rank1d_compute(self._X, algorithm=algorithm)
        self._cache[key] = df
        return df

    def rank2d(self, *, algorithm: str = "pearson") -> pl.DataFrame:
        """Pairwise feature ranking — long-form correlation matrix.

        ``algorithm`` in ``{"pearson", "spearman", "kendall", "covariance"}``.
        ``"kendall"`` routes through ``ferrum._core.kendall_tau_b``
        (Knight's O(n log n)).

        Output schema (``SCHEMA_RANK2D``): ``feature_x: Utf8``,
        ``feature_y: Utf8``, ``correlation: Float64`` — one row per
        ordered pair of features, p × p rows total.
        """
        from ..stats import rank2d_compute

        key = self._cache_key("rank2d", algorithm=algorithm)
        if key in self._cache:
            return self._cache[key]
        df = rank2d_compute(self._X, algorithm=algorithm)
        self._cache[key] = df
        return df

