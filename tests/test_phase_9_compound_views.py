"""Phase 9 compound view + Repeat sentinel tests."""
import pytest
import polars as pl

import ferrum as fe
from ferrum import Repeat
from ferrum.repeat import _RepeatPlaceholder


class TestRepeatSentinel:
    def test_column_row_layer_are_distinct_values(self):
        assert Repeat.column is not Repeat.row
        assert Repeat.row is not Repeat.layer
        assert Repeat.column.field == "column"
        assert Repeat.row.field == "row"
        assert Repeat.layer.field == "layer"

    def test_repr_is_descriptive(self):
        assert repr(Repeat.column) == "Repeat.column"
        assert repr(Repeat.row) == "Repeat.row"

    def test_singleton_identity_across_imports(self):
        from ferrum.repeat import Repeat as Repeat2
        assert Repeat.column is Repeat2.column

    def test_immutable(self):
        with pytest.raises((AttributeError, TypeError)):
            Repeat.column.field = "row"  # type: ignore

    def test_used_in_encode_serializes_as_dollar_repeat(self):
        sentinel = Repeat.column
        assert sentinel.to_repeat_dict() == {"$repeat": "column"}

    def test_chart_encode_accepts_repeat_placeholder(self):
        df = pl.DataFrame({"a": [1.0]})
        chart = fe.Chart(df).mark_point().encode(x=Repeat.column, y=Repeat.row)
        x_ch = chart._encoding["x"]
        y_ch = chart._encoding["y"]
        assert isinstance(x_ch.field, _RepeatPlaceholder)
        assert x_ch.field == Repeat.column
        assert y_ch.field == Repeat.row


@pytest.fixture
def df_xy():
    return pl.DataFrame({"x": [1.0, 2.0, 3.0, 4.0], "y": [10.0, 20.0, 30.0, 40.0]})


class TestJointChart:
    def test_construction_minimal(self, df_xy):
        center = fe.Chart(df_xy).mark_point().encode(x="x", y="y")
        jc = fe.JointChart(center)
        assert jc.center is center
        assert jc.top is None
        assert jc.right is None
        assert jc.ratio == 5
        assert jc.spacing == 10.0

    def test_construction_with_marginals(self, df_xy):
        center = fe.Chart(df_xy).mark_point().encode(x="x", y="y")
        top = fe.Chart(df_xy).mark_histogram().encode(x="x")
        right = fe.Chart(df_xy).mark_histogram().encode(x="y")
        jc = fe.JointChart(center, top=top, right=right, ratio=4, spacing=0.05)
        assert jc.top is top
        assert jc.right is right
        assert jc.ratio == 4

    def test_charts_property_filters_none(self, df_xy):
        center = fe.Chart(df_xy).mark_point().encode(x="x", y="y")
        top = fe.Chart(df_xy).mark_histogram().encode(x="x")
        jc = fe.JointChart(center, top=top, right=None)
        assert jc.charts == [center, top]
        assert len(jc.charts) == 2

    def test_theme_propagates(self, df_xy):
        center = fe.Chart(df_xy).mark_point().encode(x="x", y="y")
        top = fe.Chart(df_xy).mark_histogram().encode(x="x")
        jc = fe.JointChart(center, top=top)
        themed = jc.theme(fe.themes.default)
        assert themed.center._theme is fe.themes.default
        assert themed.top._theme is fe.themes.default
        assert jc.center._theme is None

    def test_spec_shape(self, df_xy):
        center = fe.Chart(df_xy).mark_point().encode(x="x", y="y")
        top = fe.Chart(df_xy).mark_histogram().encode(x="x")
        jc = fe.JointChart(center, top=top, ratio=3)
        spec = jc.spec
        assert spec["kind"] == "joint"
        assert "center" in spec
        assert spec["top"] is not None
        assert spec["right"] is None
        assert spec["ratio"] == 3
        assert spec["share"]["x"] == ["center", "top"]
        assert spec["share"]["y"] == ["center"]

    def test_show_svg_renders_after_task_12(self, df_xy):
        center = fe.Chart(df_xy).mark_point().encode(x="x", y="y")
        top = fe.Chart(df_xy).mark_histogram().encode(x="x")
        jc = fe.JointChart(center, top=top)
        out = jc.show_svg()
        assert out.startswith("<svg")
        assert "</svg>" in out

    def test_invalid_ratio_errors(self, df_xy):
        center = fe.Chart(df_xy).mark_point().encode(x="x", y="y")
        with pytest.raises(ValueError, match="ratio"):
            fe.JointChart(center, ratio=0)


class TestRepeatChart:
    @pytest.fixture
    def iris_like(self):
        return pl.DataFrame({
            "sepal_length": [5.1, 4.9, 4.7, 5.0, 5.4],
            "sepal_width":  [3.5, 3.0, 3.2, 3.6, 3.9],
            "petal_length": [1.4, 1.4, 1.3, 1.4, 1.7],
            "species":      ["a", "a", "b", "b", "a"],
        })

    def test_construction_minimal(self, iris_like):
        template = fe.Chart(iris_like).mark_point().encode(x=Repeat.column, y=Repeat.row)
        rc = fe.RepeatChart(template, row=["sepal_length"], column=["sepal_width"])
        assert rc.row == ["sepal_length"]
        assert rc.column == ["sepal_width"]
        assert rc.diagonal is None
        assert rc.corner is False

    def test_construction_with_diagonal_and_corner(self, iris_like):
        off = fe.Chart(iris_like).mark_point().encode(x=Repeat.column, y=Repeat.row)
        diag = fe.Chart(iris_like).mark_histogram().encode(x=Repeat.column)
        rc = fe.RepeatChart(off, row=["a", "b"], column=["a", "b"], diagonal=diag, corner=True)
        assert rc.diagonal is diag
        assert rc.corner is True

    def test_diagonal_without_row_or_column_errors(self, iris_like):
        off = fe.Chart(iris_like).mark_point().encode(x=Repeat.column, y=Repeat.row)
        diag = fe.Chart(iris_like).mark_histogram().encode(x=Repeat.column)
        with pytest.raises(ValueError, match="diagonal"):
            fe.RepeatChart(off, row=["a", "b"], diagonal=diag)

    def test_expand_2x2_yields_4_charts(self, iris_like):
        off = fe.Chart(iris_like).mark_point().encode(x=Repeat.column, y=Repeat.row)
        rc = fe.RepeatChart(off, row=["sepal_length", "sepal_width"],
                            column=["sepal_length", "sepal_width"])
        cells = rc.expand()
        assert len(cells) == 4
        for row_field, col_field, chart in cells:
            x_enc = chart._encoding.get("x")
            y_enc = chart._encoding.get("y")
            assert x_enc.field == col_field
            assert y_enc.field == row_field

    def test_expand_with_diagonal_uses_diag_for_matching_cells(self, iris_like):
        off = fe.Chart(iris_like).mark_point().encode(x=Repeat.column, y=Repeat.row)
        diag = fe.Chart(iris_like).mark_histogram().encode(x=Repeat.column)
        rc = fe.RepeatChart(off, row=["a", "b"], column=["a", "b"], diagonal=diag)
        cells = rc.expand()
        for row_field, col_field, chart in cells:
            if row_field == col_field:
                # mark_histogram is a pending stat mark; resolve to bar at to_spec time.
                # The chart's _mark may be "bar" (after pending resolve) or a placeholder.
                # Either way: the diagonal cell should have a different mark than the off-diagonal point.
                assert chart._pending_stat_mark is not None or chart._mark in ("bar",)
            else:
                assert chart._mark == "point"

    def test_expand_with_corner_filters_to_lower_triangle(self, iris_like):
        off = fe.Chart(iris_like).mark_point().encode(x=Repeat.column, y=Repeat.row)
        rc = fe.RepeatChart(off, row=["a", "b", "c"], column=["a", "b", "c"], corner=True)
        cells = rc.expand()
        coords = [(r, c) for r, c, _ in cells]
        row_idx = {"a": 0, "b": 1, "c": 2}
        for r, c in coords:
            assert row_idx[r] >= row_idx[c]

    def test_diagonal_with_asymmetric_warns(self, iris_like):
        off = fe.Chart(iris_like).mark_point().encode(x=Repeat.column, y=Repeat.row)
        diag = fe.Chart(iris_like).mark_histogram().encode(x=Repeat.column)
        with pytest.warns(UserWarning, match="diagonal"):
            fe.RepeatChart(off, row=["a", "b"], column=["x", "y"], diagonal=diag).expand()

    def test_spec_shape(self, iris_like):
        off = fe.Chart(iris_like).mark_point().encode(x=Repeat.column, y=Repeat.row)
        rc = fe.RepeatChart(off, row=["a"], column=["b"], corner=False)
        spec = rc.spec
        assert spec["kind"] == "repeat"
        assert spec["row"] == ["a"]
        assert spec["column"] == ["b"]
        assert spec["corner"] is False


class TestClusterMapChart:
    @pytest.fixture
    def df_matrix(self):
        return pl.DataFrame({
            "row_id": [0, 1, 2, 3, 4],
            "a": [1.0, 2.0, 3.0, 4.0, 5.0],
            "b": [5.0, 4.0, 3.0, 2.0, 1.0],
            "c": [1.0, 1.0, 5.0, 5.0, 1.0],
        })

    def test_construction_heatmap_only(self, df_matrix):
        heat = fe.Chart(df_matrix).mark_rect().encode(x="a", y="row_id", fill="b")
        cm = fe.ClusterMapChart(heat)
        assert cm.heatmap is heat
        assert cm.row_dendrogram is None
        assert cm.col_dendrogram is None
        assert cm.dendrogram_ratio == 0.2

    def test_charts_filters_none(self, df_matrix):
        heat = fe.Chart(df_matrix).mark_rect().encode(x="a", y="row_id", fill="b")
        col_d = fe.Chart(df_matrix).mark_rule().encode(x="a", y="b")
        cm = fe.ClusterMapChart(heat, col_dendrogram=col_d)
        assert cm.charts == [heat, col_d]

    def test_spec_shape(self, df_matrix):
        heat = fe.Chart(df_matrix).mark_rect().encode(x="a", y="row_id", fill="b")
        cm = fe.ClusterMapChart(heat, dendrogram_ratio=0.3)
        spec = cm.spec
        assert spec["kind"] == "cluster_map"
        assert "heatmap" in spec
        assert spec["row_dendrogram"] is None
        assert spec["dendrogram_ratio"] == 0.3

    def test_invalid_ratio_errors(self, df_matrix):
        heat = fe.Chart(df_matrix).mark_rect().encode(x="a", y="row_id", fill="b")
        with pytest.raises(ValueError, match="dendrogram_ratio"):
            fe.ClusterMapChart(heat, dendrogram_ratio=0.0)
        with pytest.raises(ValueError, match="dendrogram_ratio"):
            fe.ClusterMapChart(heat, dendrogram_ratio=1.5)

    def test_theme_propagates(self, df_matrix):
        heat = fe.Chart(df_matrix).mark_rect().encode(x="a", y="row_id", fill="b")
        col_d = fe.Chart(df_matrix).mark_rule().encode(x="a", y="b")
        cm = fe.ClusterMapChart(heat, col_dendrogram=col_d)
        themed = cm.theme(fe.themes.default)
        assert themed.heatmap._theme is fe.themes.default
        assert themed.col_dendrogram._theme is fe.themes.default
