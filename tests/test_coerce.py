import numpy as np
import polars as pl
import pyarrow as pa
import pytest

from ferrum._coerce import to_arrow_table


def test_polars_dataframe_zero_copy():
    df = pl.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})
    tbl = to_arrow_table(df)
    assert isinstance(tbl, pa.Table)
    assert tbl.num_rows == 3
    assert tbl.column_names == ["a", "b"]


def test_pyarrow_table_passthrough():
    tbl_in = pa.table({"x": [1, 2], "y": ["a", "b"]})
    tbl_out = to_arrow_table(tbl_in)
    assert tbl_out is tbl_in


def test_pyarrow_recordbatch_converted_to_table():
    rb = pa.RecordBatch.from_pylist([{"a": 1}, {"a": 2}])
    tbl = to_arrow_table(rb)
    assert isinstance(tbl, pa.Table)
    assert tbl.num_rows == 2


def test_dict_of_arrays():
    tbl = to_arrow_table({"a": [1, 2, 3], "b": [4, 5, 6]})
    assert isinstance(tbl, pa.Table)
    assert tbl.num_rows == 3
    assert tbl.column_names == ["a", "b"]


def test_list_of_records():
    tbl = to_arrow_table([{"a": 1, "b": 4}, {"a": 2, "b": 5}])
    assert isinstance(tbl, pa.Table)
    assert tbl.num_rows == 2


def test_numpy_2d_with_auto_column_names():
    arr = np.array([[1, 2], [3, 4], [5, 6]])
    tbl = to_arrow_table(arr)
    assert isinstance(tbl, pa.Table)
    assert tbl.column_names == ["col_0", "col_1"]
    assert tbl.num_rows == 3


def test_numpy_1d_raises_clear_typeerror():
    arr = np.array([1, 2, 3])
    with pytest.raises(TypeError, match="1D numpy arrays need column names"):
        to_arrow_table(arr)


def test_none_raises_value_error():
    with pytest.raises(ValueError, match="per-layer data"):
        to_arrow_table(None)


def test_pandas_via_narwhals():
    pd = pytest.importorskip("pandas")
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})
    tbl = to_arrow_table(df)
    assert isinstance(tbl, pa.Table)
    assert tbl.num_rows == 3


def test_unsupported_type_raises_clear_typeerror():
    class WeirdData:
        pass
    with pytest.raises(TypeError, match="Unsupported data type"):
        to_arrow_table(WeirdData())
