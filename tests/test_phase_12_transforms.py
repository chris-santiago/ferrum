"""Tests for Phase 12 data transform constructors and Chart.transform() wiring."""

import json

import polars as pl
import pytest

import ferrum as fm
from ferrum.transforms import (
    transform_aggregate,
    transform_bin,
    transform_calculate,
    transform_density,
    transform_filter,
    transform_flatten,
    transform_fold,
    transform_impute,
    transform_join_aggregate,
    transform_loess,
    transform_pivot,
    transform_regression,
    transform_sample,
    transform_stack,
    transform_timeunit,
    transform_top_k,
    transform_window,
)


# ---------------------------------------------------------------------------
# Constructor dict shape tests
# ---------------------------------------------------------------------------


class TestTransformFilter:
    def test_string_predicate(self):
        result = transform_filter("datum.x > 5")
        assert result == {"type": "filter", "predicate": "datum.x > 5"}

    def test_dict_predicate_equality(self):
        result = transform_filter({"category": "A"})
        assert result["type"] == "filter"
        assert "datum.category" in result["predicate"]

    def test_dict_predicate_list(self):
        result = transform_filter({"category": ["A", "B"]})
        assert result["type"] == "filter"
        assert "indexof" in result["predicate"]


class TestTransformCalculate:
    def test_basic(self):
        result = transform_calculate("z", "datum.x * 2")
        assert result == {
            "type": "calculate",
            "as_field": "z",
            "expr": "datum.x * 2",
        }


class TestTransformAggregate:
    def test_basic(self):
        agg = {"field": "price", "fn": "mean", "as": "avg_price"}
        result = transform_aggregate(agg, groupby=["category"])
        assert result["type"] == "data_aggregate"
        assert result["aggregates"] == [agg]
        assert result["groupby"] == ["category"]

    def test_no_groupby(self):
        agg = {"field": "x", "fn": "sum", "as": "total"}
        result = transform_aggregate(agg)
        assert "groupby" not in result


class TestTransformBin:
    def test_defaults(self):
        result = transform_bin("price")
        assert result == {"type": "data_bin", "field": "price", "nice": True}

    def test_with_options(self):
        result = transform_bin("price", as_="price_bin", maxbins=20, step=5.0, nice=False)
        assert result["type"] == "data_bin"
        assert result["field"] == "price"
        assert result["as_"] == "price_bin"
        assert result["maxbins"] == 20
        assert result["step"] == 5.0
        assert result["nice"] is False


class TestTransformFold:
    def test_defaults(self):
        result = transform_fold(["a", "b", "c"])
        assert result == {
            "type": "fold",
            "fields": ["a", "b", "c"],
            "as_": ["key", "value"],
        }

    def test_custom_as(self):
        result = transform_fold(["a", "b"], as_=("variable", "measurement"))
        assert result["as_"] == ["variable", "measurement"]


class TestTransformPivot:
    def test_defaults(self):
        result = transform_pivot("category", "amount")
        assert result == {
            "type": "pivot",
            "field": "category",
            "value": "amount",
            "op": "sum",
        }

    def test_with_options(self):
        result = transform_pivot("cat", "val", groupby=["id"], limit=10, op="mean")
        assert result["groupby"] == ["id"]
        assert result["limit"] == 10
        assert result["op"] == "mean"


class TestTransformJoinAggregate:
    def test_basic(self):
        agg = {"field": "x", "fn": "mean", "as": "x_mean"}
        result = transform_join_aggregate(agg, groupby=["g"])
        assert result["type"] == "join_aggregate"
        assert result["aggregates"] == [agg]
        assert result["groupby"] == ["g"]

    def test_no_groupby(self):
        agg = {"field": "x", "fn": "count", "as": "n"}
        result = transform_join_aggregate(agg)
        assert "groupby" not in result


class TestTransformWindow:
    def test_basic(self):
        op = {"op": "row_number", "as": "rank"}
        result = transform_window(op, sort=["score"])
        assert result["type"] == "data_window"
        assert result["ops"] == [op]
        assert result["sort"] == ["score"]

    def test_with_frame(self):
        op = {"op": "sum", "field": "x", "as": "rolling_sum"}
        result = transform_window(op, frame=(-2, 2), groupby=["g"])
        assert result["frame"] == [-2, 2]
        assert result["groupby"] == ["g"]

    def test_defaults(self):
        op = {"op": "rank", "as": "r"}
        result = transform_window(op)
        assert "sort" not in result
        assert "groupby" not in result
        assert "frame" not in result


class TestTransformDensity:
    def test_defaults(self):
        result = transform_density("x")
        assert result["type"] == "density_data"
        assert result["field"] == "x"
        assert result["cumulative"] is False
        assert result["as_"] == ["value", "density"]

    def test_with_options(self):
        result = transform_density(
            "x", bandwidth=0.5, groupby=["g"], extent=(0.0, 10.0), steps=200, cumulative=True
        )
        assert result["bandwidth"] == 0.5
        assert result["groupby"] == ["g"]
        assert result["extent"] == [0.0, 10.0]
        assert result["steps"] == 200
        assert result["cumulative"] is True


class TestTransformRegression:
    def test_defaults(self):
        result = transform_regression("x", "y")
        assert result == {
            "type": "regression_data",
            "x": "x",
            "y": "y",
            "method": "linear",
            "order": 1,
            "as_": ["x", "y"],
        }

    def test_with_options(self):
        result = transform_regression("a", "b", method="poly", order=3, groupby=["g"])
        assert result["method"] == "poly"
        assert result["order"] == 3
        assert result["groupby"] == ["g"]


class TestTransformLoess:
    def test_defaults(self):
        result = transform_loess("x", "y")
        assert result == {
            "type": "loess_data",
            "x": "x",
            "y": "y",
            "bandwidth": 0.3,
            "as_": ["x", "y"],
        }

    def test_with_options(self):
        result = transform_loess("a", "b", bandwidth=0.5, groupby=["g"])
        assert result["bandwidth"] == 0.5
        assert result["groupby"] == ["g"]


class TestTransformImpute:
    def test_defaults(self):
        result = transform_impute("x")
        assert result == {"type": "impute", "field": "x", "method": "value"}

    def test_with_options(self):
        result = transform_impute("x", method="mean", groupby=["g"], key="time")
        assert result["method"] == "mean"
        assert result["groupby"] == ["g"]
        assert result["key"] == "time"

    def test_with_value(self):
        result = transform_impute("x", value=0.0)
        assert result["value"] == 0.0


class TestTransformFlatten:
    def test_basic(self):
        result = transform_flatten(["tags"])
        assert result == {"type": "flatten", "fields": ["tags"]}

    def test_with_as(self):
        result = transform_flatten(["tags", "scores"], as_=["tag", "score"])
        assert result["as_"] == ["tag", "score"]


class TestTransformSample:
    def test_basic(self):
        result = transform_sample(100)
        assert result == {"type": "sample", "n": 100, "seed": 42}

    def test_custom_seed(self):
        result = transform_sample(50, seed=123)
        assert result["seed"] == 123


class TestTransformTopK:
    def test_defaults(self):
        result = transform_top_k(5, field="revenue")
        assert result == {
            "type": "top_k",
            "n": 5,
            "field": "revenue",
            "op": "sum",
            "sort": "descending",
        }

    def test_with_options(self):
        result = transform_top_k(3, field="score", op="mean", sort="ascending")
        assert result["op"] == "mean"
        assert result["sort"] == "ascending"


class TestTransformStack:
    def test_defaults(self):
        result = transform_stack("value", groupby=["x"])
        assert result == {
            "type": "data_stack",
            "field": "value",
            "groupby": ["x"],
            "as_": ["y0", "y1"],
            "offset": "zero",
        }

    def test_with_options(self):
        result = transform_stack(
            "value",
            groupby=["x", "color"],
            sort=["color"],
            as_=("start", "end"),
            offset="normalize",
        )
        assert result["sort"] == ["color"]
        assert result["as_"] == ["start", "end"]
        assert result["offset"] == "normalize"


class TestTransformTimeunit:
    def test_defaults(self):
        result = transform_timeunit("date", "month")
        assert result == {
            "type": "time_unit",
            "field": "date",
            "unit": "month",
            "utc": False,
        }

    def test_with_options(self):
        result = transform_timeunit("ts", "year", utc=True, as_="year_ts")
        assert result["utc"] is True
        assert result["as_"] == "year_ts"


# ---------------------------------------------------------------------------
# Chart.transform() wiring tests
# ---------------------------------------------------------------------------


class TestChartTransformWiring:
    @pytest.fixture()
    def sample_df(self):
        return pl.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0], "y": [10.0, 20.0, 30.0, 40.0, 50.0]})

    def test_transform_stores_dicts(self, sample_df):
        chart = (
            fm.Chart(sample_df)
            .mark_point()
            .encode(x="x", y="y")
            .transform(transform_filter("datum.x > 2"))
        )
        # The chart's internal transforms list should have one entry
        assert len(chart._transforms) == 1
        assert chart._transforms[0]["type"] == "filter"

    def test_transform_accumulates(self, sample_df):
        chart = (
            fm.Chart(sample_df)
            .mark_point()
            .encode(x="x", y="y")
            .transform(transform_filter("datum.x > 2"))
            .transform(transform_calculate("z", "datum.x * 2"))
        )
        assert len(chart._transforms) == 2
        assert chart._transforms[0]["type"] == "filter"
        assert chart._transforms[1]["type"] == "calculate"

    def test_transform_serializes_to_json(self, sample_df):
        chart = (
            fm.Chart(sample_df)
            .mark_point()
            .encode(x="x", y="y")
            .transform(transform_filter("datum.x > 2"))
        )
        json_str = chart.to_json()
        parsed = json.loads(json_str)
        assert "transforms" in parsed
        assert parsed["transforms"][0]["type"] == "filter"
        assert parsed["transforms"][0]["predicate"] == "datum.x > 2"

    def test_transform_immutability(self, sample_df):
        base = fm.Chart(sample_df).mark_point().encode(x="x", y="y")
        with_filter = base.transform(transform_filter("datum.x > 2"))
        assert len(base._transforms) == 0
        assert len(with_filter._transforms) == 1


# ---------------------------------------------------------------------------
# Integration test — render with a data transform
# ---------------------------------------------------------------------------


class TestTransformIntegration:
    def test_filter_renders(self):
        """A chart with transform_filter should render without error."""
        df = pl.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0], "y": [10.0, 20.0, 30.0, 40.0, 50.0]})
        chart = (
            fm.Chart(df)
            .mark_point()
            .encode(x="x", y="y")
            .transform(transform_filter("datum.x > 2"))
        )
        svg = chart.show_svg()
        assert "<svg" in svg
        # Should have fewer points rendered (only x > 2 passes)
        # Basic sanity: SVG should contain circle elements for points
        assert "circle" in svg or "<path" in svg

    def test_calculate_renders(self):
        """A chart with transform_calculate should render without error."""
        df = pl.DataFrame({"x": [1.0, 2.0, 3.0], "y": [10.0, 20.0, 30.0]})
        chart = (
            fm.Chart(df)
            .mark_point()
            .encode(x="x", y="z")
            .transform(transform_calculate("z", "datum.y * 2"))
        )
        svg = chart.show_svg()
        assert "<svg" in svg

    def test_fold_renders(self):
        """A chart with transform_fold should render without error."""
        df = pl.DataFrame({"id": [1, 2, 3], "a": [10.0, 20.0, 30.0], "b": [5.0, 15.0, 25.0]})
        chart = (
            fm.Chart(df)
            .mark_point()
            .encode(x="key", y="value")
            .transform(transform_fold(["a", "b"]))
        )
        svg = chart.show_svg()
        assert "<svg" in svg

    def test_sample_renders(self):
        """A chart with transform_sample should render without error."""
        df = pl.DataFrame({"x": list(range(100)), "y": [float(i) for i in range(100)]})
        chart = fm.Chart(df).mark_point().encode(x="x:Q", y="y:Q").transform(transform_sample(10))
        svg = chart.show_svg()
        assert "<svg" in svg


# ---------------------------------------------------------------------------
# Importability from top-level
# ---------------------------------------------------------------------------


class TestImports:
    def test_all_transforms_importable(self):
        """All 17 transform constructors are importable from ferrum."""
        assert hasattr(fm, "transform_filter")
        assert hasattr(fm, "transform_calculate")
        assert hasattr(fm, "transform_aggregate")
        assert hasattr(fm, "transform_bin")
        assert hasattr(fm, "transform_fold")
        assert hasattr(fm, "transform_pivot")
        assert hasattr(fm, "transform_join_aggregate")
        assert hasattr(fm, "transform_window")
        assert hasattr(fm, "transform_density")
        assert hasattr(fm, "transform_regression")
        assert hasattr(fm, "transform_loess")
        assert hasattr(fm, "transform_impute")
        assert hasattr(fm, "transform_flatten")
        assert hasattr(fm, "transform_sample")
        assert hasattr(fm, "transform_top_k")
        assert hasattr(fm, "transform_stack")
        assert hasattr(fm, "transform_timeunit")
