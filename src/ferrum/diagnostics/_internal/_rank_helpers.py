"""Thin wrappers around Rust rank1d/rank2d for the raw-data path.

When the caller passes a DataFrame (not a ModelSource), these helpers coerce
it to Arrow and delegate to ``ferrum._core.py_rank1d`` / ``py_rank2d``.

``polars_or_array_to_record_batch`` handles arbitrary external inputs
(polars, pandas, numpy, pyarrow).  It is the raw-data counterpart to
``BaseSource._x_record_batch``, which only handles ``self._X`` (a polars
DataFrame, or ``None`` on an X-less source -- guarded via ``_require_X``)
and is therefore simpler and faster.
"""

from __future__ import annotations

import polars as pl
import pyarrow as pa


def polars_or_array_to_record_batch(X) -> pa.RecordBatch:
    """Coerce polars DataFrame / pandas / pyarrow / 2D numpy to a pyarrow RecordBatch.

    Used by the raw-data (non-ModelSource) paths of ``rank1d_compute``,
    ``rank1d_compute_with_y``, and ``rank2d_compute``.  Internal ModelSource
    methods should call ``self._x_record_batch(diagnostic)`` instead, which
    resolves ``self._X`` through ``_require_X`` (raising a typed error on an
    X-less source) and avoids the type-dispatch overhead once it's a known
    polars DataFrame.
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


def _y_to_float64_arrow(y) -> pa.Array:
    """Coerce ``y`` to an Arrow ``Float64`` array, agreeing with ``ModelSource``.

    Mirrors the conversion ``ModelSource.rank1d`` already uses for its own
    ``y`` (``BaseSource._coerce_X_y`` wraps ``y`` in a ``polars.Series``;
    ``RankingMixin.rank1d`` then calls ``pa.array(self._y.to_list(),
    type=pa.float64())``). Building a ``polars.Series`` first -- rather than
    ``np.asarray(y, dtype=np.float64)`` -- preserves missing values as Arrow
    validity-bitmap nulls instead of silently casting them to ``NaN``, so
    this raw-data path now produces the exact same Arrow array
    ``ModelSource.rank1d`` would build from the same ``y``: the two call
    sites no longer drift (a ``NaN`` used to be indistinguishable from a
    real floating-point value once it reached the Rust kernel, corrupting
    every score; a numeric-looking string ``y`` used to be silently parsed
    here via ``numpy`` while ``ModelSource`` raised). It also means a
    non-numeric ``y`` now raises the same ``pyarrow.lib.ArrowInvalid`` the
    ``ModelSource`` path already raises, instead of ``numpy``'s more
    permissive string parsing.

    This does **not** make the covariance statistic itself null-aware.
    ``py_rank1d_with_y`` (``crates/ferrum-core/src/transform/stats/mod.rs``,
    ``py_rank1d_with_y`` / around L402-407) reads the Arrow array's raw
    value buffer and never consults the validity bitmap, so a null ``y``
    slot is read as the buffer's default ``0.0`` and folded into the
    covariance as a zero-valued observation rather than dropped or
    otherwise excluded -- on **both** the raw-data and ``ModelSource``
    arms, since both now build the same null-preserving Arrow array and
    feed the same kernel. This is pre-existing Rust-kernel behavior, not
    introduced here; it is tracked as a separate Rust-side follow-up
    (`py_rank1d_with_y` should decide and implement an explicit null
    policy -- e.g. drop-row -- rather than silently zero-impute), not
    something this Python-side coercion fix can or should change.
    """
    y_ser = y if isinstance(y, pl.Series) else pl.Series("y", y)
    return pa.array(y_ser.to_list(), type=pa.float64())


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

    batch = polars_or_array_to_record_batch(X)
    y_arrow = _y_to_float64_arrow(y)
    result = py_rank1d_with_y(batch, y_arrow, algorithm, top_k)
    return pl.from_arrow(result)


def rank2d_compute(X, *, algorithm: str = "pearson") -> pl.DataFrame:
    from ferrum._core import py_rank2d

    batch = polars_or_array_to_record_batch(X)
    result = py_rank2d(batch, algorithm)
    return pl.from_arrow(result)
