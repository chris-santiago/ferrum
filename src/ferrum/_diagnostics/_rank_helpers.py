"""Thin wrappers around Rust rank1d/rank2d for the raw-data path.

When the caller passes a DataFrame (not a ModelSource), these helpers coerce
it to Arrow and delegate to ``ferrum._core.py_rank1d`` / ``py_rank2d``.
"""

from __future__ import annotations

import polars as pl
import pyarrow as pa


def _coerce_to_arrow_batch(X) -> pa.RecordBatch:
    """Coerce polars DataFrame / pandas / 2D numpy to a pyarrow RecordBatch."""
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

    batch = _coerce_to_arrow_batch(X)
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

    batch = _coerce_to_arrow_batch(X)
    y_arrow = pa.array(np.asarray(y, dtype=np.float64), type=pa.float64())
    result = py_rank1d_with_y(batch, y_arrow, algorithm, top_k)
    return pl.from_arrow(result)


def rank2d_compute(X, *, algorithm: str = "pearson") -> pl.DataFrame:
    from ferrum._core import py_rank2d

    batch = _coerce_to_arrow_batch(X)
    result = py_rank2d(batch, algorithm)
    return pl.from_arrow(result)
