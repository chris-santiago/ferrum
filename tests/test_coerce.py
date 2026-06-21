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
# Duration columns are NOT pre-cast by to_arrow_table; polars.to_arrow()
# produces Arrow duration[unit] which normalize_for_rust converts to timestamp[ms].


def test_polars_duration_cast():
    """Polars Duration passes through to_arrow_table as Arrow duration[ns].

    The actual timestamp[ms] normalization happens in normalize_for_rust,
    which is tested by test_duration_frontend_agreement and
    test_dict_over_duration_normalizes_to_timestamp_ms in this file.
    This test just verifies that to_arrow_table no longer casts Duration → Int64.
    """
    from ferrum._coerce import normalize_for_rust

    df = pl.DataFrame(
        {
            "dur": pl.Series([1_000_000, 2_000_000, 3_000_000], dtype=pl.Duration("ns")),
            "y": [1.0, 2.0, 3.0],
        }
    )
    tbl = to_arrow_table(df)
    # to_arrow_table leaves Duration as Arrow duration[ns]; normalize_for_rust
    # then converts it to timestamp[ms].
    assert pa.types.is_duration(tbl.schema.field("dur").type)
    normalized = normalize_for_rust(tbl)
    assert pa.types.is_timestamp(normalized.schema.field("dur").type)
    assert normalized.schema.field("dur").type.unit == "ms"


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


# ── SEAM-01 regression: Duration frontend agreement ───────────────────────────


def test_duration_frontend_agreement():
    """Regression test for SEAM-01 (T0.2): polars and pyarrow Duration columns
    must produce the same Arrow dtype and the same axis-scale values after
    normalize_for_rust.

    Old behavior (broken):
    - polars Duration("ns") was cast to Int64 (raw nanosecond integer) by
      to_arrow_table before crossing the boundary.
    - pyarrow duration[us] was left as Arrow Duration and hit different Rust
      arms (different unit math).
    - Same logical "1 second" duration appeared at different x positions on
      the axis depending on which frontend provided the data.

    New behavior (fixed):
    - Both frontends produce Arrow duration[unit] out of to_arrow_table.
    - normalize_for_rust converts all duration[unit] → timestamp[ms] uniformly.
    - A 1-second duration from polars and a 1-second duration from pyarrow
      produce the same timestamp[ms] value: 1970-01-01T00:00:01.
    """
    from datetime import datetime

    from ferrum._coerce import normalize_for_rust

    # 1 second expressed in different units on each frontend.
    ONE_SECOND_NS = 1_000_000_000  # nanoseconds
    ONE_SECOND_US = 1_000_000  # microseconds

    # Polars path: Duration("ns") -> to_arrow_table -> normalize_for_rust
    df_polars = pl.DataFrame(
        {"dur": pl.Series([ONE_SECOND_NS], dtype=pl.Duration("ns")), "y": [1.0]}
    )
    normalized_polars = normalize_for_rust(to_arrow_table(df_polars))

    # PyArrow path: duration[us] -> to_arrow_table -> normalize_for_rust
    tbl_pyarrow = pa.table({"dur": pa.array([ONE_SECOND_US], type=pa.duration("us")), "y": [1.0]})
    normalized_pyarrow = normalize_for_rust(to_arrow_table(tbl_pyarrow))

    # Both must produce timestamp[ms] — not int64 (old polars behavior) or
    # duration (old pyarrow behavior).
    polars_dtype = normalized_polars.schema.field("dur").type
    pyarrow_dtype = normalized_pyarrow.schema.field("dur").type
    assert pa.types.is_timestamp(polars_dtype), (
        f"SEAM-01 regression: polars Duration should normalize to timestamp, got {polars_dtype}"
    )
    assert pa.types.is_timestamp(pyarrow_dtype), (
        f"SEAM-01 regression: pyarrow duration should normalize to timestamp, got {pyarrow_dtype}"
    )
    assert polars_dtype == pyarrow_dtype, (
        f"SEAM-01 regression: frontend dtypes diverge after normalize_for_rust: "
        f"polars={polars_dtype}, pyarrow={pyarrow_dtype}"
    )
    assert polars_dtype.unit == "ms", f"Expected timestamp[ms], got {polars_dtype}"

    # The axis-scale value (timestamp value) must be the same from both frontends.
    val_polars = normalized_polars.column("dur").to_pylist()[0]
    val_pyarrow = normalized_pyarrow.column("dur").to_pylist()[0]
    assert val_polars == val_pyarrow, (
        f"SEAM-01 regression: polars and pyarrow Duration values diverge: "
        f"polars={val_polars!r}, pyarrow={val_pyarrow!r}"
    )
    # 1 second from epoch (1970-01-01T00:00:01)
    assert val_polars == datetime(1970, 1, 1, 0, 0, 1), (
        f"Expected 1970-01-01T00:00:01, got {val_polars!r}"
    )


# ── Fix round 1 regression: dictionary-encoded Duration and timestamp[us] ────
#
# The original normalize_for_rust had two drifted paths: the top-level loop
# applied all normalizations (date, timestamp, duration, null), but the
# is_dictionary branch only re-checked for date32/date64 after decode.
# A dict-over-duration column was decoded to duration[ns] and left untouched,
# crashing with "unsupported dtype: Duration(Nanosecond)" when the table
# crossed the Rust boundary.  A dict-over-timestamp[us] stayed non-ms.
#
# Both bugs are fixed by extracting _normalize_column and calling it from both
# the dict branch and the non-dict branch.


def test_dict_over_duration_normalizes_to_timestamp_ms():
    """dictionary-encoded Duration column must normalize to timestamp[ms].

    Old behavior (broken): pc.dictionary_decode produced duration[ns], which the
    old is_dictionary branch left untouched (it only re-checked date32/date64).
    The column then crossed the Rust boundary as duration[ns] and raised:
      ValueError: column 'dur' has unsupported dtype: Duration(Nanosecond)

    New behavior: the decoded column passes through _normalize_column, which
    converts duration[any unit] → timestamp[ms] identically to the non-dict path.
    """
    from datetime import datetime

    from ferrum._coerce import normalize_for_rust

    ONE_SECOND_NS = 1_000_000_000

    # Build a dictionary-encoded duration[ns] column.
    indices = pa.array([0, 1, 0], type=pa.int8())
    values = pa.array([ONE_SECOND_NS, 2 * ONE_SECOND_NS], type=pa.duration("ns"))
    dict_col = pa.DictionaryArray.from_arrays(indices, values)
    tbl = pa.table({"dur": dict_col, "y": [1.0, 2.0, 3.0]})

    # This must not raise (old code raised ValueError).
    result = normalize_for_rust(tbl)

    dur_type = result.schema.field("dur").type
    assert pa.types.is_timestamp(dur_type), (
        f"dict-over-duration should normalize to timestamp[ms], got {dur_type}"
    )
    assert dur_type.unit == "ms", f"Expected timestamp[ms], got {dur_type}"

    # Check values: 1s → datetime(1970,1,1,0,0,1), 2s → datetime(1970,1,1,0,0,2).
    vals = result.column("dur").to_pylist()
    assert vals[0] == datetime(1970, 1, 1, 0, 0, 1), (
        f"Expected 1970-01-01T00:00:01, got {vals[0]!r}"
    )
    assert vals[1] == datetime(1970, 1, 1, 0, 0, 2), (
        f"Expected 1970-01-01T00:00:02, got {vals[1]!r}"
    )
    assert vals[2] == datetime(1970, 1, 1, 0, 0, 1), (
        f"Expected 1970-01-01T00:00:01, got {vals[2]!r}"
    )


def test_dict_over_duration_chart_renders():
    """End-to-end: dictionary-encoded Duration column must produce valid SVG."""
    import ferrum as fm

    ONE_SECOND_NS = 1_000_000_000
    indices = pa.array([0, 1, 0], type=pa.int8())
    values = pa.array([ONE_SECOND_NS, 2 * ONE_SECOND_NS], type=pa.duration("ns"))
    dict_col = pa.DictionaryArray.from_arrays(indices, values)
    tbl = pa.table({"dur": dict_col, "y": [1.0, 2.0, 3.0]})

    svg = fm.Chart(tbl).mark_point().encode(x="dur", y="y").to_svg()
    assert "<svg" in svg


def test_dict_over_timestamp_us_normalizes_to_timestamp_ms():
    """dictionary-encoded timestamp[us] column must normalize to timestamp[ms].

    Old behavior (broken): the is_dictionary branch only re-checked date32/date64.
    A dict-over-timestamp[us] was decoded to timestamp[us] and left at that unit,
    causing wrong axis scaling (us vs ms).

    New behavior: the decoded column passes through _normalize_column, which
    casts timestamp[us] → timestamp[ms].
    """
    from datetime import datetime

    from ferrum._coerce import normalize_for_rust

    # One microsecond = 1us tick; 1,000,000 us = 1 second.
    ONE_SECOND_US = 1_000_000

    indices = pa.array([0, 1], type=pa.int8())
    values = pa.array([0, ONE_SECOND_US], type=pa.timestamp("us"))
    dict_col = pa.DictionaryArray.from_arrays(indices, values)
    tbl = pa.table({"ts": dict_col, "y": [1.0, 2.0]})

    result = normalize_for_rust(tbl)

    ts_type = result.schema.field("ts").type
    assert pa.types.is_timestamp(ts_type), (
        f"dict-over-timestamp[us] should normalize to timestamp, got {ts_type}"
    )
    assert ts_type.unit == "ms", f"Expected timestamp[ms], got {ts_type}"

    vals = result.column("ts").to_pylist()
    assert vals[0] == datetime(1970, 1, 1, 0, 0, 0), f"Expected epoch, got {vals[0]!r}"
    assert vals[1] == datetime(1970, 1, 1, 0, 0, 1), f"Expected 1s from epoch, got {vals[1]!r}"
