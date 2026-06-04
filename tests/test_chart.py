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
    # The shorthand desugars into a _PendingAggregate sentinel; the sentinel is
    # resolved into a Rust Aggregate object with inferred groupby at to_spec() time.
    from ferrum.encoding.base import _PendingAggregate

    assert any(isinstance(t, _PendingAggregate) for t in c._transforms)


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
    assert '"x":' in j


def test_chart_data_input_pyarrow_table():
    tbl = pa.table({"a": [1, 2], "b": [3, 4]})
    c = Chart(tbl).mark_point().encode(x="a", y="b")
    # to_svg actually exercises the coerce path; smoke-test only here
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
        Chart(arr).mark_point().to_svg()  # to_svg triggers coerce


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


# ---- BUG-3: __add__ always layers (redesigned) ----


def test_add_differing_data_produces_layered_chart():
    """+ with different data auto null-pad merges and layers."""
    df1 = pl.DataFrame({"x": [1, 2], "y": [3, 4]})
    df2 = pl.DataFrame({"x": [5, 6], "y": [7, 8]})
    c1 = Chart(df1).mark_point().encode(x="x", y="y")
    c2 = Chart(df2).mark_line().encode(x="x", y="y")
    result = c1 + c2
    # Always a layered Chart, never HConcatChart
    assert isinstance(result, Chart)
    assert result._layers is not None
    assert len(result._layers) == 2


def test_add_differing_columns_null_pad_merges():
    """+ with disjoint columns produces unified data with all columns."""
    df1 = pl.DataFrame({"a": [1, 2], "b": [3, 4]})
    df2 = pl.DataFrame({"c": [5, 6], "d": [7, 8]})
    c1 = Chart(df1).mark_point().encode(x="a", y="b")
    c2 = Chart(df2).mark_line().encode(x="c", y="d")
    result = c1 + c2
    unified = result._data
    assert isinstance(unified, pl.DataFrame)
    assert set(unified.columns) == {"a", "b", "c", "d"}
    # Original rows from df1 have nulls in c, d columns
    assert unified["c"][:2].null_count() == 2
    # Original rows from df2 have nulls in a, b columns
    assert unified["a"][2:].null_count() == 2


# ---- BUG-5: mark_smooth transforms lost when layered via + ----


def test_layered_smooth_renders_paths():
    """points + smooth via + must render <path> elements for the trend line.

    Regression: _expand_layers placed single-mark transforms into _Layer.transforms
    instead of top-level transforms.  The Rust renderer only executes spec.transforms
    (chart-level); layer.transforms is stored but never executed, so the Smooth
    output batch was never produced and zero <path> elements appeared in the SVG.
    """
    df = pl.DataFrame(
        {
            "x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            "y": [2.1, 3.9, 6.0, 8.2, 10.1, 12.0, 13.8, 16.1, 18.0, 20.2],
        }
    )
    points = Chart(df).mark_point().encode(x="x", y="y")
    trend = Chart(df).mark_smooth(method="lm").encode(x="x", y="y")
    chart = points + trend
    svg = chart.to_svg()
    # ferrum renders line marks as <polyline>, not <path>.
    line_count = svg.count("<polyline")
    assert line_count > 0, (
        f"Expected at least one <polyline> element for the smooth trend line, got 0.\n"
        f"The Smooth transform on the trend layer was silently dropped."
    )


# ---- BUG-5b: scatter layer reads smooth output instead of original data ----


def test_layered_smooth_scatter_count_matches_original_data():
    """Scatter circle count must equal the original row count, not the smooth grid size.

    The Smooth transform outputs 200 evenly-spaced grid points by default.
    When points + smooth is composed, the scatter layer must read from the
    original data (60 rows), not from FINAL_OUTPUT_KEY after the smooth runs.
    """
    import numpy as np

    rng = np.random.default_rng(0)
    n = 60
    df = pl.DataFrame(
        {
            "x": np.linspace(1.0, 10.0, n),
            "y": np.linspace(2.0, 12.0, n) + rng.normal(0, 0.8, n),
        }
    )
    points = Chart(df).mark_point().encode(x="x", y="y")
    trend = Chart(df).mark_smooth(method="lm").encode(x="x", y="y")
    svg = (points + trend).to_svg()
    circle_count = svg.count("<circle")
    assert circle_count == n, (
        f"Expected {n} scatter circles (original row count), got {circle_count}.\n"
        f"Scatter layer is reading smooth-curve output instead of original data."
    )


def test_layered_smooth_real_column_names_do_not_crash():
    """Composing scatter + smooth with non-x/y column names must not raise.

    Previously, the smooth transform replaced FINAL_OUTPUT_KEY with a batch
    whose columns are 'x', 'y', 'ci_lower', 'ci_upper'.  The scatter encoding
    referenced 'sepal_length' / 'petal_length', which don't exist in that
    output — causing a ValueError at render time.
    """
    import numpy as np

    rng = np.random.default_rng(0)
    n = 50
    df = pl.DataFrame(
        {
            "sepal_length": np.linspace(4.0, 8.0, n),
            "petal_length": np.linspace(1.0, 7.0, n) + rng.normal(0, 0.3, n),
        }
    )
    points = Chart(df).mark_point().encode(x="sepal_length", y="petal_length")
    trend = Chart(df).mark_smooth(method="lm").encode(x="sepal_length", y="petal_length")
    svg = (points + trend).to_svg()  # must not raise
    assert svg.count("<circle") == n, f"Expected {n} scatter circles, got {svg.count('<circle')}."
    assert svg.count("<polyline") > 0, "Expected at least one smooth trend line."


def test_layered_smooth_grouped_by_color_correct_counts():
    """With color grouping, scatter count = total rows; line count = number of groups.

    150 rows across 3 species → 150 scatter circles + 3 smooth polylines.
    """
    import numpy as np

    rng = np.random.default_rng(0)
    n_per_group = 50
    df = pl.DataFrame(
        {
            "x": np.tile(np.linspace(1.0, 10.0, n_per_group), 3),
            "y": np.concatenate(
                [
                    np.linspace(1.0, 10.0, n_per_group) + rng.normal(0, 0.5, n_per_group),
                    np.linspace(2.0, 14.0, n_per_group) + rng.normal(0, 0.5, n_per_group),
                    np.linspace(0.5, 6.0, n_per_group) + rng.normal(0, 0.5, n_per_group),
                ]
            ),
            "species": ["setosa"] * n_per_group
            + ["versicolor"] * n_per_group
            + ["virginica"] * n_per_group,
        }
    )
    total = n_per_group * 3
    points = Chart(df).mark_point().encode(x="x", y="y", color="species:N")
    # groupby must be set explicitly on mark_smooth for per-species fitting.
    trend = (
        Chart(df)
        .mark_smooth(method="lm", groupby="species")
        .encode(x="x", y="y", color="species:N")
    )
    svg = (points + trend).to_svg()
    # The SVG also contains 3 legend circles (one per group), so total
    # circles = data circles + legend circles.
    n_groups = 3
    assert svg.count("<circle") == total + n_groups, (
        f"Expected {total + n_groups} circles ({total} data + {n_groups} legend), "
        f"got {svg.count('<circle')}."
    )
    assert svg.count("<polyline") == n_groups, (
        f"Expected {n_groups} smooth lines (one per species), got {svg.count('<polyline')}."
    )


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
