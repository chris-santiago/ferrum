"""Phase 10g — feature ranking (rank1d, rank2d)."""

from __future__ import annotations

import polars as pl
import pyarrow as pa


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
        key = self._cache_key("rank1d", algorithm=algorithm)
        if key in self._cache:
            return self._cache[key]

        x_arrow = pa.RecordBatch.from_pydict({c: self._X[c].to_arrow() for c in self._X.columns})
        if algorithm == "covariance":
            from ferrum._core import py_rank1d_with_y

            if self._y is None:
                raise ValueError("ModelSource.rank1d(algorithm='covariance') requires y.")
            y_arrow = pa.array(self._y.to_list(), type=pa.float64())
            result = py_rank1d_with_y(x_arrow, y_arrow, algorithm, None)
        else:
            from ferrum._core import py_rank1d

            result = py_rank1d(x_arrow, algorithm, None)
        df = pl.from_arrow(result)
        self._cache[key] = df
        return df

    def rank2d(self, *, algorithm: str = "pearson") -> pl.DataFrame:
        """Pairwise feature ranking — long-form correlation matrix.

        ``algorithm`` in ``{"pearson", "spearman", "kendall", "covariance"}``.
        All algorithms now run in Rust (Kendall uses Knight's O(n log n)).

        Output schema (``SCHEMA_RANK2D``): ``feature_x: Utf8``,
        ``feature_y: Utf8``, ``correlation: Float64`` — one row per
        ordered pair of features, p × p rows total.
        """
        from ferrum._core import py_rank2d

        key = self._cache_key("rank2d", algorithm=algorithm)
        if key in self._cache:
            return self._cache[key]
        x_arrow = pa.RecordBatch.from_pydict({c: self._X[c].to_arrow() for c in self._X.columns})
        result = py_rank2d(x_arrow, algorithm)
        df = pl.from_arrow(result)
        self._cache[key] = df
        return df
