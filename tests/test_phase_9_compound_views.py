"""Phase 9 compound view + Repeat sentinel tests."""

import pytest
import polars as pl

import ferrum as fe
from ferrum import Repeat
from ferrum.repeat import _RepeatPlaceholder
from tests._svg_extents import x_axis_extents, y_axis_extents


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

    def test_to_svg_renders_after_task_12(self, df_xy):
        center = fe.Chart(df_xy).mark_point().encode(x="x", y="y")
        top = fe.Chart(df_xy).mark_histogram().encode(x="x")
        jc = fe.JointChart(center, top=top)
        out = jc.to_svg()
        assert out.startswith("<svg")
        assert "</svg>" in out

    def test_invalid_ratio_errors(self, df_xy):
        center = fe.Chart(df_xy).mark_point().encode(x="x", y="y")
        with pytest.raises(ValueError, match="ratio"):
            fe.JointChart(center, ratio=0)


class TestRepeatChart:
    @pytest.fixture
    def iris_like(self):
        return pl.DataFrame(
            {
                "sepal_length": [5.1, 4.9, 4.7, 5.0, 5.4],
                "sepal_width": [3.5, 3.0, 3.2, 3.6, 3.9],
                "petal_length": [1.4, 1.4, 1.3, 1.4, 1.7],
                "species": ["a", "a", "b", "b", "a"],
            }
        )

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
        rc = fe.RepeatChart(
            off, row=["sepal_length", "sepal_width"], column=["sepal_length", "sepal_width"]
        )
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

    def test_diagonal_with_asymmetric_raises(self, iris_like):
        off = fe.Chart(iris_like).mark_point().encode(x=Repeat.column, y=Repeat.row)
        diag = fe.Chart(iris_like).mark_histogram().encode(x=Repeat.column)
        with pytest.raises(ValueError, match="row == column"):
            fe.RepeatChart(off, row=["a", "b"], column=["x", "y"], diagonal=diag).expand()

    def test_spec_shape(self, iris_like):
        off = fe.Chart(iris_like).mark_point().encode(x=Repeat.column, y=Repeat.row)
        rc = fe.RepeatChart(off, row=["a"], column=["b"], corner=False)
        spec = rc.spec
        assert spec["kind"] == "repeat"
        assert spec["row"] == ["a"]
        assert spec["column"] == ["b"]
        assert spec["corner"] is False

    def test_no_axes_set_errors(self, iris_like):
        off = fe.Chart(iris_like).mark_point()
        with pytest.raises(ValueError, match="at least one of row"):
            fe.RepeatChart(off)

    def test_corner_without_grid_errors(self, iris_like):
        off = fe.Chart(iris_like).mark_point().encode(x=Repeat.column)
        with pytest.raises(ValueError, match="corner"):
            fe.RepeatChart(off, column=["a", "b"], corner=True)

    def test_columns_must_be_positive(self, iris_like):
        off = fe.Chart(iris_like).mark_point().encode(x=Repeat.column)
        with pytest.raises(ValueError, match="columns"):
            fe.RepeatChart(off, column=["a"], columns=0)

    def test_column_only_wrap_1d(self, iris_like):
        off = fe.Chart(iris_like).mark_point().encode(x=Repeat.column, y="sepal_width")
        rc = fe.RepeatChart(off, column=["sepal_length", "sepal_width", "petal_length"])
        cells = rc.expand()
        assert len(cells) == 3
        for row_field, col_field, chart in cells:
            assert row_field is None
            assert chart._encoding["x"].field == col_field

    def test_row_only_wrap_1d(self, iris_like):
        off = fe.Chart(iris_like).mark_point().encode(x="sepal_length", y=Repeat.row)
        rc = fe.RepeatChart(off, row=["sepal_width", "petal_length"])
        cells = rc.expand()
        assert len(cells) == 2
        for row_field, col_field, chart in cells:
            assert col_field is None
            assert chart._encoding["y"].field == row_field

    def test_column_wrap_dimensions(self, iris_like):
        off = fe.Chart(iris_like).mark_point().encode(x=Repeat.column, y="sepal_width")
        # 5 cells, columns=2 → 3 rows × 2 cols
        rc = fe.RepeatChart(off, column=["a", "b", "c", "d", "e"], columns=2)
        n_cols, n_rows = rc._wrap_dimensions(len(rc.expand()))
        assert (n_cols, n_rows) == (2, 3)

    def test_column_wrap_default_single_row(self, iris_like):
        off = fe.Chart(iris_like).mark_point().encode(x=Repeat.column, y="sepal_width")
        rc = fe.RepeatChart(off, column=["a", "b", "c"])
        n_cols, n_rows = rc._wrap_dimensions(len(rc.expand()))
        assert (n_cols, n_rows) == (3, 1)

    def test_row_wrap_default_single_column(self, iris_like):
        off = fe.Chart(iris_like).mark_point().encode(x="sepal_length", y=Repeat.row)
        rc = fe.RepeatChart(off, row=["a", "b", "c"])
        n_cols, n_rows = rc._wrap_dimensions(len(rc.expand()))
        assert (n_cols, n_rows) == (1, 3)

    def test_template_uses_missing_axis_raises(self, iris_like):
        # template references Repeat.row, but only column= is set
        off = fe.Chart(iris_like).mark_point().encode(x=Repeat.column, y=Repeat.row)
        rc = fe.RepeatChart(off, column=["sepal_length", "sepal_width"])
        with pytest.raises(ValueError, match="Repeat.row but row="):
            rc.expand()

    def test_layer_only_produces_single_layered_cell(self, iris_like):
        # layer-only repeat: one cell, N layers stacked.
        tpl = fe.Chart(iris_like).mark_point().encode(x="sepal_length", y=Repeat.layer)
        rc = fe.RepeatChart(tpl, layer=["sepal_width", "petal_length"])
        cells = rc.expand()
        assert len(cells) == 1
        row_field, col_field, chart = cells[0]
        assert row_field is None and col_field is None
        # Two layers: one for each layer field. _layers stores them.
        assert chart._layers is not None
        assert len(chart._layers) == 2

    def test_layer_with_column_produces_layered_cells(self, iris_like):
        # column + layer: 1-D wrap, each cell layered over the layer fields.
        tpl = fe.Chart(iris_like).mark_point().encode(x=Repeat.column, y=Repeat.layer)
        rc = fe.RepeatChart(
            tpl, column=["sepal_length", "sepal_width"], layer=["petal_length", "species"]
        )
        cells = rc.expand()
        assert len(cells) == 2
        for row_field, col_field, chart in cells:
            assert row_field is None
            assert chart._layers is not None
            assert len(chart._layers) == 2

    def test_layer_with_grid_produces_layered_cells(self, iris_like):
        tpl = (
            fe.Chart(iris_like)
            .mark_point()
            .encode(x=Repeat.column, y=Repeat.row, color=Repeat.layer)
        )
        rc = fe.RepeatChart(
            tpl,
            row=["sepal_length", "sepal_width"],
            column=["sepal_length", "sepal_width"],
            layer=["species", "petal_length"],
        )
        cells = rc.expand()
        assert len(cells) == 4
        for row_field, col_field, chart in cells:
            assert chart._layers is not None
            assert len(chart._layers) == 2

    def test_layer_without_setting_raises(self, iris_like):
        # template references Repeat.layer but layer= not set
        tpl = fe.Chart(iris_like).mark_point().encode(x="sepal_length", y=Repeat.layer)
        rc = fe.RepeatChart(tpl, column=["sepal_length"])
        with pytest.raises(ValueError, match="Repeat.layer but layer="):
            rc.expand()

    def test_resolve_must_be_dict(self, iris_like):
        tpl = fe.Chart(iris_like).mark_point().encode(x=Repeat.column, y="sepal_width")
        with pytest.raises(ValueError, match="resolve must be a dict"):
            fe.RepeatChart(tpl, column=["sepal_length"], resolve="shared")

    def test_resolve_rejects_unknown_mode(self, iris_like):
        tpl = fe.Chart(iris_like).mark_point().encode(x=Repeat.column, y="sepal_width")
        with pytest.raises(ValueError, match="expected 'shared' or 'independent'"):
            fe.RepeatChart(tpl, column=["sepal_length"], resolve={"x": "smart"})

    def test_resolve_independent_is_noop(self, iris_like):
        tpl = fe.Chart(iris_like).mark_point().encode(x=Repeat.column, y="sepal_width")
        rc_a = fe.RepeatChart(tpl, column=["sepal_length", "sepal_width"])
        rc_b = fe.RepeatChart(
            tpl,
            column=["sepal_length", "sepal_width"],
            resolve={"x": "independent"},
        )
        # The encodings of both cell sets must match — independent is the
        # default behavior and resolve= should not have rewritten any scale.
        cells_a = rc_a.expand()
        cells_b = rc_b.expand()
        for (_, _, ca), (_, _, cb) in zip(cells_a, cells_b):
            assert ca._encoding["x"].field == cb._encoding["x"].field
            assert ca._encoding["x"]._kwargs == cb._encoding["x"]._kwargs

    def test_resolve_shared_x_injects_union_domain(self):
        # Two columns with disjoint ranges; resolve={"x": "shared"} must
        # union the x domain across panels at RENDER time -- the composite
        # tree's Rust resolve pass -- not by injecting a scale= dict into
        # expand()'s materialized panels (expand() is a pre-render
        # materialization step; resolve= is render-time only).
        df = pl.DataFrame(
            {
                "small": [0.0, 1.0, 2.0, 3.0, 4.0],
                "big": [10.0, 12.0, 14.0, 18.0, 20.0],
                "y": [1, 2, 3, 4, 5],
            }
        )
        tpl = fe.Chart(df).mark_point().encode(x=Repeat.column, y="y")
        rc = fe.RepeatChart(tpl, column=["small", "big"], resolve={"x": "shared"})
        cells = rc.expand()
        assert len(cells) == 2
        for _, _, chart in cells:
            assert chart._encoding["x"]._kwargs.get("scale") is None

        svg = rc.to_svg()
        extents = x_axis_extents(svg)
        assert len(extents) == 2
        assert extents[0] == extents[1]
        assert extents[0].hi >= 20.0

    def test_resolve_shared_y_with_layered_cells(self):
        # Each cell stacks two layers, each binding a different field on y.
        # The shared-y domain must span ALL layer fields, not just the
        # first -- proven by the rendered tick extent, since expand() no
        # longer injects a scale= dict for resolve= sharing.
        df = pl.DataFrame(
            {
                "x": [1, 2, 3, 4, 5],
                "y1": [1.0, 2.0, 3.0, 4.0, 5.0],
                "y2": [50.0, 60.0, 70.0, 80.0, 90.0],
            }
        )
        tpl = fe.Chart(df).mark_line().encode(x="x", y=Repeat.layer)
        rc = fe.RepeatChart(tpl, layer=["y1", "y2"], resolve={"y": "shared"})
        cells = rc.expand()
        assert len(cells) == 1
        _, _, chart = cells[0]
        assert chart._layers is not None
        for layer in chart._layers:
            assert layer.encoding["y"]._kwargs.get("scale") is None

        svg = rc.to_svg()
        extents = y_axis_extents(svg)
        assert len(extents) == 1
        assert extents[0].lo <= 10.0
        assert extents[0].hi >= 90.0

    def test_resolve_shared_categorical_color(self):
        df = pl.DataFrame(
            {
                "x": [1, 2, 3, 4],
                "c1": ["a", "b", "a", "b"],
                "c2": ["b", "c", "c", "a"],
            }
        )
        tpl = fe.Chart(df).mark_point().encode(x="x", y="x", color=Repeat.column)
        rc = fe.RepeatChart(tpl, column=["c1", "c2"], resolve={"color": "shared"})
        cells = rc.expand()
        for _, _, chart in cells:
            # expand() never injects scale= -- resolve= sharing rides the
            # composite tree's resolve field, not a per-panel scale= dict.
            assert chart._encoding["color"]._kwargs.get("scale") is None

        # The tree resolve field is what actually drives the Rust
        # composite resolve pass's cross-panel ordinal union (see
        # test_composite_render_repeat_layer.py for the render-side proof
        # of that pass with numeric/categorical channels).
        lowered = rc._composite_tree(auto_tooltips=False)
        assert lowered.tree["resolve"] == {"color": "shared"}


class TestClusterMapChart:
    @pytest.fixture
    def df_matrix(self):
        return pl.DataFrame(
            {
                "row_id": [0, 1, 2, 3, 4],
                "a": [1.0, 2.0, 3.0, 4.0, 5.0],
                "b": [5.0, 4.0, 3.0, 2.0, 1.0],
                "c": [1.0, 1.0, 5.0, 5.0, 1.0],
            }
        )

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


class TestShareScale:
    """``_ChartLike.share_scale(**channels)`` — K16 / P2.8."""

    def test_rejects_unknown_mode(self):
        df = pl.DataFrame({"x": [1, 2, 3], "y": [1, 2, 3]})
        c = fe.Chart(df).mark_point().encode(x="x", y="y")
        with pytest.raises(ValueError, match="expected 'shared' or 'independent'"):
            (c | c).share_scale(x="smart")

    def test_no_shared_channels_returns_self(self):
        df = pl.DataFrame({"x": [1, 2, 3], "y": [1, 2, 3]})
        c = fe.Chart(df).mark_point().encode(x="x", y="y")
        comp = c | c
        # No channels given at all is a true no-op (identity).
        assert comp.share_scale() is comp
        # An explicit "independent" is meaningful resolve= metadata now (no
        # longer a silent identity no-op) even though it renders the same as
        # the (also-independent) default.
        result = comp.share_scale(x="independent")
        assert result is not comp
        assert result._resolve == {"x": "independent"}
        assert result.to_svg() == comp.to_svg()

    def test_hconcat_share_x(self):
        df = pl.DataFrame(
            {
                "a": [0.0, 1.0, 2.0, 3.0],
                "b": [10.0, 20.0, 30.0, 40.0],
                "y": [1, 2, 3, 4],
            }
        )
        left = fe.Chart(df).mark_point().encode(x="a", y="y")
        right = fe.Chart(df).mark_point().encode(x="b", y="y")
        shared = (left | right).share_scale(x="shared")
        assert shared._resolve == {"x": "shared"}
        # share_scale never injects scale= -- resolve= sharing is render-time.
        for child in shared.charts:
            assert child._encoding["x"]._kwargs.get("scale") is None
        # Must render byte-identical to the equivalent resolve= construction.
        equivalent = fe.HConcatChart([left, right], resolve={"x": "shared"})
        assert shared.to_svg() == equivalent.to_svg()
        extents = x_axis_extents(shared.to_svg())
        assert len(extents) == 2
        assert extents[0] == extents[1]
        assert extents[0].hi >= 40.0

    def test_vconcat_share_y(self):
        df = pl.DataFrame(
            {
                "x": [1, 2, 3, 4],
                "low": [0.0, 1.0, 2.0, 3.0],
                "high": [100.0, 200.0, 300.0, 400.0],
            }
        )
        top = fe.Chart(df).mark_point().encode(x="x", y="low")
        bot = fe.Chart(df).mark_point().encode(x="x", y="high")
        shared = (top & bot).share_scale(y="shared")
        assert shared._resolve == {"y": "shared"}
        for child in shared.charts:
            assert child._encoding["y"]._kwargs.get("scale") is None
        equivalent = fe.VConcatChart([top, bot], resolve={"y": "shared"})
        assert shared.to_svg() == equivalent.to_svg()
        # Sharing must have actually changed the render relative to the
        # independent default, proving the resolve field drove a real union.
        assert shared.to_svg() != (top & bot).to_svg()

    def test_joint_share_y_raises_unsupported(self):
        # JointChart carries no resolve= field at all -- its marginal
        # alignment is fixed grid geometry, not a resolve= channel. share_scale
        # must raise a typed error instead of silently performing the old
        # (now-removed) raw-column scale injection.
        df = pl.DataFrame(
            {
                "x": [1.0, 2.0, 3.0, 4.0],
                "y": [10.0, 50.0, 90.0, 30.0],
                "y_marginal": [5.0, 25.0, 75.0, 100.0],
            }
        )
        center = fe.Chart(df).mark_point().encode(x="x", y="y")
        right = fe.Chart(df).mark_point().encode(x="x", y="y_marginal")
        joint = fe.JointChart(center, right=right)
        with pytest.raises(ValueError, match="JointChart.*resolve="):
            joint.share_scale(y="shared")

    def test_clustermap_share_x_raises_unsupported(self):
        # Same reasoning as JointChart above: ClusterMapChart has no
        # resolve= field, so share_scale must raise rather than diverge.
        df = pl.DataFrame({"a": [1.0, 2.0, 3.0], "b": [10.0, 20.0, 30.0], "r": [0, 1, 2]})
        heat = fe.Chart(df).mark_rect().encode(x="a", y="r", fill="b")
        col_d = fe.Chart(df).mark_rule().encode(x="b", y="r")
        cm = fe.ClusterMapChart(heat, col_dendrogram=col_d)
        with pytest.raises(ValueError, match="ClusterMapChart.*resolve="):
            cm.share_scale(x="shared")

    def test_repeat_share_scale_merges_into_resolve(self):
        df = pl.DataFrame(
            {
                "small": [0.0, 1.0, 2.0],
                "big": [10.0, 20.0, 30.0],
                "y": [1, 2, 3],
            }
        )
        tpl = fe.Chart(df).mark_point().encode(x=Repeat.column, y="y")
        rc = fe.RepeatChart(tpl, column=["small", "big"])
        shared = rc.share_scale(x="shared")
        assert shared.resolve == {"x": "shared"}
        # expand() no longer injects scale= -- sharing rides the composite
        # tree's resolve field at render time.
        cells = shared.expand()
        for _, _, chart in cells:
            assert chart._encoding["x"]._kwargs.get("scale") is None
        extents = x_axis_extents(shared.to_svg())
        assert len(extents) == 2
        assert extents[0] == extents[1]
        assert extents[0].hi >= 30.0

    def test_repeat_share_scale_merges_with_existing_resolve(self):
        df = pl.DataFrame(
            {
                "small": [0.0, 1.0, 2.0],
                "big": [10.0, 20.0, 30.0],
                "y": [1, 2, 3],
            }
        )
        tpl = fe.Chart(df).mark_point().encode(x=Repeat.column, y="y")
        rc = fe.RepeatChart(tpl, column=["small", "big"], resolve={"y": "shared"})
        shared = rc.share_scale(x="shared")
        assert shared.resolve == {"y": "shared", "x": "shared"}

    def test_share_scale_silent_when_channel_unbound(self):
        # Two charts neither of which binds 'size' — share_scale should
        # silently leave them alone (no scale injection, no exception).
        df = pl.DataFrame({"x": [1, 2, 3], "y": [1, 2, 3]})
        c = fe.Chart(df).mark_point().encode(x="x", y="y")
        shared = (c | c).share_scale(size="shared")
        for child in shared.charts:
            assert "size" not in child._encoding
