"""Arrow-CDI transport bridge between the Python API and the Rust core."""

from __future__ import annotations

from typing import Any

from ferrum._core import process_batch as _process_batch


def process_batch(data: Any) -> Any:
    """Pass an Arrow-compatible object through the Rust pipeline.

    Accepts any object implementing __arrow_c_stream__:
    polars DataFrame, pyarrow Table, pyarrow RecordBatch, etc.
    Returns a PyRecordBatchReader consumable via:
        polars  — pl.from_arrow(result)
        pyarrow — pa.RecordBatchReader.from_stream(result).read_all()
    """
    if not hasattr(data, "__arrow_c_stream__"):
        raise TypeError(
            f"Expected an Arrow-compatible object (polars DataFrame, "
            f"pyarrow Table/RecordBatch), got {type(data).__name__!r}"
        )
    return _process_batch(data)
