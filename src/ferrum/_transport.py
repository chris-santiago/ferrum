"""Arrow-CDI transport bridge between the Python API and the Rust core."""

from __future__ import annotations

from typing import Any

from ferrum._core import process_batch as _process_batch


def process_batch(data: Any) -> Any:
    """Pass an Arrow-compatible object through the Rust pipeline via CDI.

    Accepts any object implementing the Arrow C Data Interface
    (``__arrow_c_stream__``): polars ``DataFrame``, pyarrow ``Table``,
    pyarrow ``RecordBatch``, etc.

    Parameters
    ----------
    data : Any
        An Arrow-compatible object.  Must implement ``__arrow_c_stream__``.

    Returns
    -------
    PyRecordBatchReader
        The processed batch reader.  Consume with:

        * polars  — ``pl.from_arrow(result)``
        * pyarrow — ``pa.RecordBatchReader.from_stream(result).read_all()``

    Raises
    ------
    TypeError
        If ``data`` does not implement ``__arrow_c_stream__``.

    Examples
    --------
    >>> import pyarrow as pa
    >>> tbl = pa.table({"a": [1, 2, 3]})
    >>> result = process_batch(tbl)
    """
    if not hasattr(data, "__arrow_c_stream__"):
        raise TypeError(
            f"Expected an Arrow-compatible object (polars DataFrame, "
            f"pyarrow Table/RecordBatch), got {type(data).__name__!r}"
        )
    return _process_batch(data)
