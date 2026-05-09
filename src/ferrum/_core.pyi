from typing import Any

def process_batch(data: Any) -> Any:
    """Accept any Arrow stream (__arrow_c_stream__), apply column rename, return Arrow stream.

    Returns a PyRecordBatchReader. Consume with pl.from_arrow(result) or
    pa.Table.from_batches(list(result)).
    """
    ...
