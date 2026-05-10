import polars as pl
import pyarrow as pa
import pytest

from ferrum import Chart


def test_chart_construction_with_polars():
    df = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    c = Chart(df)
    assert c._data is df


def test_chart_immutability_mark_returns_new_chart():
    df = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    c1 = Chart(df)
    c2 = c1.mark_point()
    assert c1 is not c2
    assert c1._mark is None
    assert c2._mark == "point"


def test_chart_encode_returns_new_chart():
    df = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    c1 = Chart(df).mark_point()
    c2 = c1.encode(x="a", y="b")
    assert c1 is not c2
    assert c1._encoding == {}
    assert "x" in c2._encoding


def test_chart_mark_point_with_kwargs():
    df = pl.DataFrame({"a": [1], "b": [2]})
    c = Chart(df).mark_point(size=100, stroke="#ff0000")
    assert c._mark == "point"
    assert c._mark_kwargs == {"size": 100, "stroke": "#ff0000"}


def test_chart_encode_with_string_field():
    df = pl.DataFrame({"price": [1.0], "weight": [2.0]})
    c = Chart(df).mark_point().encode(x="price", y="weight")
    assert c._encoding["x"].field == "price"
    assert c._encoding["y"].field == "weight"


def test_chart_encode_with_shorthand_aggregate():
    df = pl.DataFrame({"price": [1.0]})
    c = Chart(df).mark_bar().encode(y="mean(price)")
    # The shorthand should desugar into an Aggregate transform
    assert any(t.__class__.__name__ == "Aggregate" for t in c._transforms)


def test_chart_encode_with_explicit_channel_class():
    from ferrum.encoding import X, Y
    df = pl.DataFrame({"a": [1], "b": [2]})
    c = Chart(df).mark_point().encode(x=X("a", type="Q"), y=Y("b"))
    assert c._encoding["x"].field == "a"
    assert c._encoding["x"]._kwargs["type"] == "Q"


def test_chart_to_spec_returns_chartspec():
    from ferrum import ChartSpec
    df = pl.DataFrame({"a": [1], "b": [2]})
    c = Chart(df).mark_point().encode(x="a", y="b")
    spec = c.to_spec()
    assert isinstance(spec, ChartSpec)
    assert spec.mark == "point"


def test_chart_to_json_round_trip():
    df = pl.DataFrame({"a": [1], "b": [2]})
    c = Chart(df).mark_point().encode(x="a", y="b")
    j = c.to_json()
    assert "point" in j
    assert "\"x\":" in j


def test_chart_data_input_pyarrow_table():
    tbl = pa.table({"a": [1, 2], "b": [3, 4]})
    c = Chart(tbl).mark_point().encode(x="a", y="b")
    # show_svg actually exercises the coerce path; smoke-test only here
    spec = c.to_spec()
    assert spec.mark == "point"


def test_chart_data_input_dict():
    c = Chart({"a": [1, 2], "b": [3, 4]}).mark_point().encode(x="a", y="b")
    assert c._mark == "point"


def test_chart_data_input_list_of_records():
    c = Chart([{"a": 1, "b": 2}, {"a": 3, "b": 4}]).mark_point().encode(x="a", y="b")
    assert c._mark == "point"


def test_chart_data_input_numpy_2d():
    np = pytest.importorskip("numpy")
    arr = np.array([[1, 2], [3, 4]])
    c = Chart(arr).mark_point().encode(x="col_0", y="col_1")
    assert c._mark == "point"


def test_chart_data_input_numpy_1d_raises():
    np = pytest.importorskip("numpy")
    arr = np.array([1, 2, 3])
    with pytest.raises(TypeError, match="1D numpy"):
        Chart(arr).mark_point().show_svg()  # show_svg triggers coerce


def test_chart_properties_sets_metadata():
    df = pl.DataFrame({"a": [1], "b": [2]})
    c = Chart(df).mark_point().properties(width=800, height=600, title="Hello")
    assert c._width == 800
    assert c._height == 600
    assert c._title == "Hello"


def test_chart_with_pandas_dataframe():
    pd = pytest.importorskip("pandas")
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    c = Chart(df).mark_point().encode(x="a", y="b")
    spec = c.to_spec()
    assert spec.mark == "point"


def test_chart_immutability_chain_independence():
    """base.encode(x='a') and base.encode(x='b') are independent."""
    df = pl.DataFrame({"a": [1], "b": [2]})
    base = Chart(df).mark_point()
    ca = base.encode(x="a")
    cb = base.encode(x="b")
    assert ca._encoding["x"].field == "a"
    assert cb._encoding["x"].field == "b"
    # base unaffected
    assert base._encoding == {}
