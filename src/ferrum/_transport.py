from __future__ import annotations

from typing import Any

from ferrum._core import process_batch as _process_batch


def process_batch(data: Any) -> Any:
    """Pass an Arrow-compatible object through the Rust pipeline.

    Accepts any object implementing __arrow_c_stream__:
    polars DataFrame, pyarrow Table, pyarrow RecordBatch, etc.
    Returns a PyRecordBatchReader. Consume with:
        polars  — pl.from_arrow(result)
        pyarrow — pa.Table.from_batches(list(result))
    """
    if not hasattr(data, "__arrow_c_stream__"):
        raise TypeError(
            f"Expected an Arrow-compatible object (polars DataFrame, "
            f"pyarrow Table/RecordBatch), got {type(data).__name__!r}"
        )
    result = _process_batch(data)

    # If PyArrow is available and input was a PyArrow type, wrap the result
    # as a PyArrow RecordBatchReader for better compatibility
    try:
        import pyarrow as pa
        # Check if input was a PyArrow type
        if isinstance(data, (pa.Table, pa.RecordBatch)):
            # Convert the arro3 RecordBatchReader to a PyArrow one
            result = pa.RecordBatchReader.from_stream(result)
    except ImportError:
        pass

    return result
