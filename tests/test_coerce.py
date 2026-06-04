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


def test_polars_categorical_cast_to_string():
    df = pl.DataFrame(
        {"x": [1.0, 2.0, 3.0], "cat": pl.Series(["a", "b", "a"], dtype=pl.Categorical)}
    )
    tbl = to_arrow_table(df)
    assert tbl.num_rows == 3
    assert pa.types.is_large_string(tbl.schema.field("cat").type) or pa.types.is_string(
        tbl.schema.field("cat").type
    )
    assert tbl.column("cat").to_pylist() == ["a", "b", "a"]


def test_polars_enum_cast_to_string():
    df = pl.DataFrame(
        {"x": [1.0, 2.0, 3.0], "grade": pl.Series(["A", "B", "A"], dtype=pl.Enum(["A", "B", "C"]))}
    )
    tbl = to_arrow_table(df)
    assert tbl.num_rows == 3
    assert pa.types.is_large_string(tbl.schema.field("grade").type) or pa.types.is_string(
        tbl.schema.field("grade").type
    )
    assert tbl.column("grade").to_pylist() == ["A", "B", "A"]


def test_polars_categorical_renders_svg():
    import ferrum as fm

    df = pl.DataFrame(
        {
            "x": [1.0, 2.0, 3.0],
            "y": [4.0, 5.0, 6.0],
            "cat": pl.Series(["a", "b", "a"], dtype=pl.Categorical),
        }
    )
    svg = fm.Chart(df).mark_point().encode(x="x", y="y", color="cat").to_svg()
    assert "<svg" in svg


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


# ── Narwhals ingestion (pandas end-to-end) ────────────────────────────────


class TestNarwhalsPandasIngestion:
    """End-to-end narwhals ingestion: pandas DataFrame → Arrow → SVG render."""

    @pytest.fixture()
    def pd(self):
        return pytest.importorskip("pandas")

    def test_pandas_int_float_columns_render_svg(self, pd):
        df = pd.DataFrame({"x": [1, 2, 3], "y": [4.0, 5.0, 6.0]})
        import ferrum as fm

        svg = fm.Chart(df).mark_point().encode(x="x", y="y").to_svg()
        assert "<svg" in svg
        assert "<circle" in svg or 'cx="' in svg

    def test_pandas_string_column_preserved(self, pd):
        df = pd.DataFrame({"x": [1, 2], "y": [3, 4], "label": ["a", "b"]})
        tbl = to_arrow_table(df)
        assert "label" in tbl.column_names
        assert tbl.column("label").to_pylist() == ["a", "b"]

    def test_pandas_empty_dataframe(self, pd):
        df = pd.DataFrame({"x": [], "y": []})
        tbl = to_arrow_table(df)
        assert isinstance(tbl, pa.Table)
        assert tbl.num_rows == 0

    def test_pandas_empty_dataframe_renders_without_error(self, pd):
        df = pd.DataFrame({"x": pd.array([], dtype="float64"), "y": pd.array([], dtype="float64")})
        import ferrum as fm

        svg = fm.Chart(df).mark_point().encode(x="x", y="y").to_svg()
        assert "<svg" in svg

    def test_pandas_single_row(self, pd):
        df = pd.DataFrame({"x": [42.0], "y": [99.0]})
        import ferrum as fm

        svg = fm.Chart(df).mark_point().encode(x="x", y="y").to_svg()
        assert "<circle" in svg or 'cx="' in svg

    def test_pandas_nullable_int(self, pd):
        df = pd.DataFrame(
            {"x": pd.array([1, 2, None], dtype="Int64"), "y": pd.array([4.0, 5.0, 6.0])}
        )
        tbl = to_arrow_table(df)
        assert tbl.num_rows == 3
        assert tbl.column("x").null_count >= 1

    def test_pandas_nullable_float(self, pd):
        df = pd.DataFrame(
            {"x": pd.array([1.0, None, 3.0], dtype="Float64"), "y": [10.0, 20.0, 30.0]}
        )
        tbl = to_arrow_table(df)
        assert tbl.num_rows == 3
        assert tbl.column("x").null_count >= 1

    def test_pandas_boolean_column(self, pd):
        df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6], "flag": [True, False, True]})
        tbl = to_arrow_table(df)
        assert tbl.column("flag").to_pylist() == [True, False, True]

    def test_pandas_categorical_column(self, pd):
        df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6], "cat": pd.Categorical(["a", "b", "a"])})
        tbl = to_arrow_table(df)
        assert tbl.num_rows == 3
        vals = tbl.column("cat").to_pylist()
        assert set(vals) == {"a", "b"}

    def test_pandas_datetime_column(self, pd):
        df = pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-01-01", "2024-06-15", "2024-12-31"]),
                "y": [1.0, 2.0, 3.0],
            }
        )
        tbl = to_arrow_table(df)
        assert tbl.num_rows == 3
        assert pa.types.is_timestamp(tbl.schema.field("date").type)

    def test_pandas_datetime_renders_svg(self, pd):
        df = pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-01-01", "2024-06-15", "2024-12-31"]),
                "y": [1.0, 2.0, 3.0],
            }
        )
        import ferrum as fm

        svg = fm.Chart(df).mark_line().encode(x="date", y="y").to_svg()
        assert "<svg" in svg

    def test_pandas_mixed_dtypes(self, pd):
        df = pd.DataFrame(
            {
                "x": [1.0, 2.0, 3.0],
                "y": [10, 20, 30],
                "name": ["alpha", "beta", "gamma"],
                "active": [True, False, True],
            }
        )
        tbl = to_arrow_table(df)
        assert tbl.num_rows == 3
        assert set(tbl.column_names) == {"x", "y", "name", "active"}

    def test_pandas_mixed_dtypes_render_svg(self, pd):
        df = pd.DataFrame(
            {
                "x": [1.0, 2.0, 3.0],
                "y": [10, 20, 30],
                "group": ["a", "a", "b"],
            }
        )
        import ferrum as fm

        svg = fm.Chart(df).mark_point().encode(x="x", y="y", color="group").to_svg()
        assert "<svg" in svg

    def test_pandas_with_nan_values(self, pd):
        df = pd.DataFrame({"x": [1.0, float("nan"), 3.0], "y": [4.0, 5.0, float("nan")]})
        tbl = to_arrow_table(df)
        assert tbl.num_rows == 3

    def test_pandas_with_nan_renders_without_crash(self, pd):
        df = pd.DataFrame({"x": [1.0, float("nan"), 3.0], "y": [4.0, 5.0, float("nan")]})
        import ferrum as fm

        svg = fm.Chart(df).mark_point().encode(x="x", y="y").to_svg()
        assert "<svg" in svg

    def test_pandas_multiindex_columns_flattened(self, pd):
        df = pd.DataFrame({"x": [1, 2], "y": [3, 4]})
        tbl = to_arrow_table(df)
        assert tbl.column_names == ["x", "y"]

    def test_pandas_column_names_preserved_exactly(self, pd):
        df = pd.DataFrame({"My Column": [1, 2], "Another Col": [3, 4]})
        tbl = to_arrow_table(df)
        assert "My Column" in tbl.column_names
        assert "Another Col" in tbl.column_names

    def test_pandas_large_dataframe_converts(self, pd):
        n = 10_000
        df = pd.DataFrame({"x": range(n), "y": range(n)})
        tbl = to_arrow_table(df)
        assert tbl.num_rows == n

    def test_pandas_data_values_roundtrip(self, pd):
        df = pd.DataFrame({"x": [1.5, 2.5, 3.5], "y": [10, 20, 30]})
        tbl = to_arrow_table(df)
        assert tbl.column("x").to_pylist() == [1.5, 2.5, 3.5]
        assert tbl.column("y").to_pylist() == [10, 20, 30]

    def test_pandas_bar_chart_renders(self, pd):
        df = pd.DataFrame({"category": ["a", "b", "c"], "value": [10, 20, 15]})
        import ferrum as fm

        svg = fm.Chart(df).mark_bar().encode(x="category", y="value").to_svg()
        assert "<svg" in svg
        assert "<rect" in svg

    def test_pandas_line_chart_renders(self, pd):
        df = pd.DataFrame({"x": [1, 2, 3, 4], "y": [1.0, 4.0, 2.0, 3.0]})
        import ferrum as fm

        svg = fm.Chart(df).mark_line().encode(x="x", y="y").to_svg()
        assert "<svg" in svg


# ── K5: Series acceptance ─────────────────────────────────────────────────────


def test_polars_series_accepted():
    s = pl.Series("x", [1.0, 2.0, 3.0])
    tbl = to_arrow_table(s)
    assert tbl.num_rows == 3
    assert tbl.column_names == ["x"]


def test_pandas_series_accepted():
    pd = pytest.importorskip("pandas")
    s = pd.Series([1.0, 2.0, 3.0], name="x")
    tbl = to_arrow_table(s)
    assert tbl.num_rows == 3
    assert "x" in tbl.column_names


def test_pandas_unnamed_series():
    pd = pytest.importorskip("pandas")
    s = pd.Series([1.0, 2.0, 3.0])
    tbl = to_arrow_table(s)
    assert tbl.num_rows == 3


# ── K6: Polars Duration cast ──────────────────────────────────────────────────


def test_polars_duration_cast():
    df = pl.DataFrame(
        {
            "dur": pl.Series([1_000_000, 2_000_000, 3_000_000], dtype=pl.Duration("ns")),
            "y": [1.0, 2.0, 3.0],
        }
    )
    tbl = to_arrow_table(df)
    assert pa.types.is_int64(tbl.schema.field("dur").type)


def test_polars_duration_renders():
    df = pl.DataFrame(
        {
            "dur": pl.Series([1_000_000, 2_000_000, 3_000_000], dtype=pl.Duration("ns")),
            "y": [1.0, 2.0, 3.0],
        }
    )
    import ferrum as fm

    svg = fm.Chart(df).mark_point().encode(x="dur", y="y").to_svg()
    assert "<svg" in svg


# ── K7: PyArrow Date32/Date64 cast ───────────────────────────────────────────


def test_pyarrow_date32_cast():
    arr = pa.array([0, 1, 100], type=pa.date32())
    tbl = pa.table({"d": arr, "y": [1.0, 2.0, 3.0]})
    result = to_arrow_table(tbl)
    assert pa.types.is_timestamp(result.schema.field("d").type)


def test_pyarrow_date64_cast():
    arr = pa.array([0, 86_400_000, 172_800_000], type=pa.date64())
    tbl = pa.table({"d": arr, "y": [1.0, 2.0, 3.0]})
    result = to_arrow_table(tbl)
    assert pa.types.is_timestamp(result.schema.field("d").type)


def test_pyarrow_date32_renders():
    import ferrum as fm

    arr = pa.array([0, 1, 100], type=pa.date32())
    tbl = pa.table({"d": arr, "y": [1.0, 2.0, 3.0]})
    svg = fm.Chart(tbl).mark_point().encode(x="d", y="y").to_svg()
    assert "<svg" in svg


def test_pyarrow_non_date_table_passthrough():
    """Non-date pyarrow tables must still pass through as the same object."""
    tbl = pa.table({"x": [1, 2, 3], "y": [4.0, 5.0, 6.0]})
    result = to_arrow_table(tbl)
    assert result is tbl
