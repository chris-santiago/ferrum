"""Data ingestion: normalize any supported input to a pyarrow.Table.

Supports (per spec §3.18):
- polars.DataFrame, polars.LazyFrame
- pyarrow.Table, pyarrow.RecordBatch
- pandas + modin + cuDF + dask + ibis (via narwhals)
- dict[str, list], list[dict]
- numpy.ndarray (2D, auto-named "col_0", "col_1", ...)

Raises TypeError for unsupported types or numpy 1D without column names.
"""
from __future__ import annotations

from typing import Any


def to_arrow_table(data: Any) -> "pyarrow.Table":
    """Normalize any supported input to a pyarrow.Table.

    Raises:
        ValueError: if data is None.
        TypeError: if input is numpy 1D, or an unsupported type.
        ImportError: if narwhals is required for the input type but not installed.
    """
    import pyarrow as pa

    if data is None:
        raise ValueError(
            "Chart(data=None) requires per-layer data — not yet supported in Phase 8a"
        )

    # Fast path: polars (zero-copy via Arrow C Data Interface)
    try:
        import polars as pl
        if isinstance(data, pl.DataFrame):
            return data.to_arrow()
        if isinstance(data, pl.LazyFrame):
            return data.collect().to_arrow()
    except ImportError:
        pass

    # Fast path: pyarrow native
    if isinstance(data, pa.Table):
        return data
    if isinstance(data, pa.RecordBatch):
        return pa.Table.from_batches([data])

    # Direct conversions: dict, list, numpy
    if isinstance(data, dict):
        return pa.Table.from_pydict(data)
    if isinstance(data, list):
        if not data:
            raise ValueError("Cannot construct Chart from empty list")
        if not isinstance(data[0], dict):
            raise TypeError(
                f"Chart(list) expects a list of dicts (one per row), got list of "
                f"{type(data[0]).__name__}"
            )
        return pa.Table.from_pylist(data)

    # numpy
    try:
        import numpy as np
        if isinstance(data, np.ndarray):
            if data.ndim == 1:
                raise TypeError(
                    "1D numpy arrays need column names — pass `Chart({'value': arr})` "
                    "or `Chart(arr.reshape(-1, 1), columns=['value'])`."
                )
            if data.ndim == 2:
                cols = [f"col_{i}" for i in range(data.shape[1])]
                return pa.table({cols[i]: data[:, i] for i in range(data.shape[1])})
            raise TypeError(f"numpy arrays with ndim={data.ndim} not supported (use 2D)")
    except ImportError:
        pass

    # Everything else: try narwhals
    try:
        import narwhals as nw
    except ImportError as e:
        raise ImportError(
            f"Input type {type(data).__name__} requires narwhals. "
            f"Install with `pip install narwhals` (or use polars/pyarrow directly)."
        ) from e

    try:
        nw_df = nw.from_native(data, eager_only=True)
        return nw_df.to_arrow()
    except (TypeError, NotImplementedError) as e:
        raise TypeError(
            f"Unsupported data type: {type(data).__name__}. "
            f"Supported: polars, pyarrow, pandas, modin, cuDF, dask, ibis, dict, list, numpy 2D. "
            f"Underlying error: {e}"
        ) from e
