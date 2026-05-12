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
    # Schwabish SB1: Chart._title is normalized to a Title value class.
    assert c._title.text == "Hello"


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


# ---- BUG-1: to_json(indent=) regression ----

def test_to_json_indent_none_is_compact():
    df = pl.DataFrame({"x": [1], "y": [2]})
    j = Chart(df).mark_point().encode(x="x", y="y").to_json()
    assert "\n" not in j


def test_to_json_indent_produces_multiline():
    df = pl.DataFrame({"x": [1], "y": [2]})
    j = Chart(df).mark_point().encode(x="x", y="y").to_json(indent=2)
    assert "\n" in j
    assert "  " in j  # indentation present


# ---- BUG-3: __add__ warning message regression ----

def test_add_differing_data_warns_with_actionable_message():
    import warnings
    df1 = pl.DataFrame({"x": [1, 2], "y": [3, 4]})
    df2 = pl.DataFrame({"x": [5, 6], "y": [7, 8]})
    c1 = Chart(df1).mark_point().encode(x="x", y="y")
    c2 = Chart(df2).mark_line().encode(x="x", y="y")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = c1 + c2
    assert len(w) == 1
    msg = str(w[0].message)
    assert "null padding" in msg
    assert "decision_boundary_chart" in msg
    # falls back to HConcatChart
    from ferrum.composition import HConcatChart
    assert isinstance(result, HConcatChart)


# ---- BUG-4: mark_shap_waterfall(sample_idx=-1) sentinel regression ----

def test_mark_shap_waterfall_default_raises_immediately():
    df = pl.DataFrame({"x": [1], "y": [2]})
    with pytest.raises(ValueError, match="sample_idx"):
        Chart(df).mark_shap_waterfall()


def test_mark_shap_waterfall_explicit_idx_does_not_raise():
    df = pl.DataFrame({"x": [1], "y": [2]})
    # Should not raise at mark time; error would only come at render time
    c = Chart(df).mark_shap_waterfall(sample_idx=0)
    assert c._pending_stat_mark is not None


# ---- BUG-5: mark_segment redundant _position assignment regression ----

def test_mark_segment_position_set_once():
    """_set_mark handles position; mark_segment must not double-assign."""
    from ferrum.position import Identity
    df = pl.DataFrame({"x": [1], "y": [2], "x2": [3], "y2": [4]})
    pos = Identity()
    c = Chart(df).mark_segment(position=pos)
    assert c._position is pos
