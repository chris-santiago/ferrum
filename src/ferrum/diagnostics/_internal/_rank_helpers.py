"""Thin wrappers around Rust rank1d/rank2d for the raw-data path.

When the caller passes a DataFrame (not a ModelSource), these helpers coerce
it to Arrow and delegate to ``ferrum._core.py_rank1d`` / ``py_rank2d``.

``polars_or_array_to_record_batch`` handles arbitrary external inputs
(polars, pandas, numpy, pyarrow).  It is the raw-data counterpart to
``BaseSource._x_record_batch``, which only handles ``self._X`` (always a
polars DataFrame) and is therefore simpler and faster.
"""

from __future__ import annotations

import polars as pl
import pyarrow as pa


def polars_or_array_to_record_batch(X) -> pa.RecordBatch:
    """Coerce polars DataFrame / pandas / pyarrow / 2D numpy to a pyarrow RecordBatch.

    Used by the raw-data (non-ModelSource) paths of ``rank1d_compute``,
    ``rank1d_compute_with_y``, and ``rank2d_compute``.  Internal ModelSource
    methods should call ``self._x_record_batch()`` instead, which assumes
    ``self._X`` is already a polars DataFrame and avoids the type-dispatch
    overhead.
    """
    if isinstance(X, pa.RecordBatch):
        return X
    if isinstance(X, pa.Table):
        return X.to_batches()[0] if X.num_rows > 0 else pa.RecordBatch.from_pydict({})
    if isinstance(X, pl.DataFrame):
        return pa.RecordBatch.from_pydict({c: X[c].to_arrow() for c in X.columns})
    if hasattr(X, "to_numpy") and hasattr(X, "columns"):
        cols = list(X.columns)
        import numpy as np

        arr = np.asarray(X, dtype=np.float64)
        return pa.RecordBatch.from_pydict(
            {str(c): pa.array(arr[:, i], type=pa.float64()) for i, c in enumerate(cols)}
        )
    import numpy as np

    arr = np.asarray(X, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"X must be 2D; got shape {arr.shape}")
    return pa.RecordBatch.from_pydict(
        {f"f{j}": pa.array(arr[:, j], type=pa.float64()) for j in range(arr.shape[1])}
    )


def rank1d_compute(
    X,
    *,
    algorithm: str = "shapiro",
    top_k: int | None = None,
) -> pl.DataFrame:
    from ferrum._core import py_rank1d

    batch = polars_or_array_to_record_batch(X)
    result = py_rank1d(batch, algorithm, top_k)
    return pl.from_arrow(result)


def rank1d_compute_with_y(
    X,
    y,
    *,
    algorithm: str = "covariance",
    top_k: int | None = None,
) -> pl.DataFrame:
    from ferrum._core import py_rank1d_with_y

    import numpy as np

    batch = polars_or_array_to_record_batch(X)
    y_arrow = pa.array(np.asarray(y, dtype=np.float64), type=pa.float64())
    result = py_rank1d_with_y(batch, y_arrow, algorithm, top_k)
    return pl.from_arrow(result)


def rank2d_compute(X, *, algorithm: str = "pearson") -> pl.DataFrame:
    from ferrum._core import py_rank2d

    batch = polars_or_array_to_record_batch(X)
    result = py_rank2d(batch, algorithm)
    return pl.from_arrow(result)
