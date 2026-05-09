import pyarrow as pa
import polars as pl
import pytest

from ferrum._transport import process_batch


def test_polars_round_trip():
    df = pl.DataFrame({"x": [1, 2, 3], "y": [4.0, 5.0, 6.0]})
    result = process_batch(df)
    out = pl.from_arrow(result)
    assert "x_renamed" in out.columns
    assert "y" in out.columns
    assert len(out) == 3


def test_pyarrow_round_trip():
    table = pa.table({"x": [1, 2, 3], "y": [4.0, 5.0, 6.0]})
    result = process_batch(table)
    out = pa.RecordBatchReader.from_stream(result).read_all()
    assert out.schema.field(0).name == "x_renamed"
    assert out.schema.field(1).name == "y"
    assert len(out) == 3


def test_pyarrow_multichunk_round_trip():
    batch1 = pa.record_batch({"x": [1, 2], "y": [3.0, 4.0]})
    batch2 = pa.record_batch({"x": [5, 6], "y": [7.0, 8.0]})
    table = pa.Table.from_batches([batch1, batch2])
    result = process_batch(table)
    out = pa.RecordBatchReader.from_stream(result).read_all()
    assert out.schema.field(0).name == "x_renamed"
    assert len(out) == 4


def test_invalid_input_raises():
    with pytest.raises(TypeError, match="Arrow-compatible"):
        process_batch({"not": "arrow"})
