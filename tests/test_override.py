"""Tests for Chart.override(), Chart.configure_*(), Chart.configure(),
and Chart.__add__ dispatch for Configure / Annotate / structural types.
"""

from __future__ import annotations

import pytest
import polars as pl

import ferrum as fm
from ferrum.configure import (
    AxisConfig,
    ColorConfig,
    Configure,
    GridConfig,
    LegendConfig,
    PaddingConfig,
    TitleConfig,
)
from ferrum.annotation.container import Annotate
from ferrum.annotation.primitives import AnnotationText
from ferrum.structural import BreakAxis, Inset, SecondaryY


# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------


@pytest.fixture()
def base_chart():
    df = pl.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
    return fm.Chart(df).mark_point().encode(x="x", y="y")


# ---------------------------------------------------------------------------
# Chart.override()
# ---------------------------------------------------------------------------


class TestOverride:
    def test_stores_kwarg(self, base_chart):
        c = base_chart.override(x_axis_label_angle=-45)
        assert c._overrides == {"x_axis_label_angle": -45}

    def test_multiple_calls_merge(self, base_chart):
        c = base_chart.override(x_axis_label_angle=-45).override(width=600)
        assert c._overrides["x_axis_label_angle"] == -45
        assert c._overrides["width"] == 600

    def test_later_call_wins_on_conflict(self, base_chart):
        c = base_chart.override(x_axis_label_angle=-45).override(x_axis_label_angle=0)
        assert c._overrides["x_axis_label_angle"] == 0

    def test_does_not_mutate_original(self, base_chart):
        _ = base_chart.override(x_axis_label_angle=-45)
        assert base_chart._overrides == {}

    def test_returns_new_chart(self, base_chart):
        c = base_chart.override(x_axis_label_angle=-45)
        assert c is not base_chart

    def test_empty_override_is_noop(self, base_chart):
        c = base_chart.override()
        assert c._overrides == {}
        assert c is not base_chart


# ---------------------------------------------------------------------------
# Chart.configure_axis()
# ---------------------------------------------------------------------------


class TestConfigureAxis:
    def test_appends_configure_layer(self, base_chart):
        c = base_chart.configure_axis(label_angle=-45)
        assert len(c._configure) == 1
        cfg = c._configure[0]
        assert isinstance(cfg, Configure)
        assert cfg.axis is not None
        assert cfg.axis.label_angle == -45

    def test_does_not_mutate_original(self, base_chart):
        _ = base_chart.configure_axis(label_angle=-45)
        assert base_chart._configure == []

    def test_returns_new_chart(self, base_chart):
        c = base_chart.configure_axis(label_angle=-45)
        assert c is not base_chart

    def test_multiple_calls_accumulate(self, base_chart):
        c = base_chart.configure_axis(label_angle=-45).configure_axis(grid=True)
        assert len(c._configure) == 2


# ---------------------------------------------------------------------------
# Chart.configure_legend()
# ---------------------------------------------------------------------------


class TestConfigureLegend:
    def test_appends_configure_layer(self, base_chart):
        c = base_chart.configure_legend(orient="bottom", columns=3)
        assert len(c._configure) == 1
        cfg = c._configure[0]
        assert cfg.legend is not None
        assert cfg.legend.orient == "bottom"
        assert cfg.legend.columns == 3

    def test_does_not_mutate_original(self, base_chart):
        _ = base_chart.configure_legend(orient="top")
        assert base_chart._configure == []


# ---------------------------------------------------------------------------
# Chart.configure_title()
# ---------------------------------------------------------------------------


class TestConfigureTitle:
    def test_appends_configure_layer(self, base_chart):
        c = base_chart.configure_title(font_size=18, anchor="start")
        assert len(c._configure) == 1
        cfg = c._configure[0]
        assert cfg.title is not None
        assert cfg.title.font_size == 18
        assert cfg.title.anchor == "start"


# ---------------------------------------------------------------------------
# Chart.configure_grid()
# ---------------------------------------------------------------------------


class TestConfigureGrid:
    def test_appends_configure_layer(self, base_chart):
        c = base_chart.configure_grid(color="#eee", width=0.5)
        assert len(c._configure) == 1
        cfg = c._configure[0]
        assert cfg.grid is not None
        assert cfg.grid.color == "#eee"
        assert cfg.grid.width == 0.5


# ---------------------------------------------------------------------------
# Chart.configure_padding()
# ---------------------------------------------------------------------------


class TestConfigurePadding:
    def test_appends_configure_layer(self, base_chart):
        c = base_chart.configure_padding(top=20, bottom=10)
        assert len(c._configure) == 1
        cfg = c._configure[0]
        assert cfg.padding is not None
        assert cfg.padding.top == 20
        assert cfg.padding.bottom == 10


# ---------------------------------------------------------------------------
# Chart.configure_color()
# ---------------------------------------------------------------------------


class TestConfigureColor:
    def test_appends_configure_layer(self, base_chart):
        c = base_chart.configure_color(scheme="tableau10")
        assert len(c._configure) == 1
        cfg = c._configure[0]
        assert cfg.color is not None
        assert cfg.color.scheme == "tableau10"


# ---------------------------------------------------------------------------
# Chart.configure() — typed config objects
# ---------------------------------------------------------------------------


class TestConfigure:
    def test_accepts_typed_objects(self, base_chart):
        c = base_chart.configure(
            axis=AxisConfig(label_angle=-45),
            legend=LegendConfig(orient="bottom"),
        )
        assert len(c._configure) == 1
        cfg = c._configure[0]
        assert cfg.axis.label_angle == -45
        assert cfg.legend.orient == "bottom"

    def test_does_not_mutate_original(self, base_chart):
        _ = base_chart.configure(axis=AxisConfig(label_angle=-45))
        assert base_chart._configure == []

    def test_returns_new_chart(self, base_chart):
        c = base_chart.configure(axis=AxisConfig())
        assert c is not base_chart

    def test_empty_configure_is_noop_payload(self, base_chart):
        # configure() with no args still appends one Configure() (all-None)
        c = base_chart.configure()
        assert len(c._configure) == 1
        assert c._configure[0] == Configure()

    def test_axis_x_y_y2_routed(self, base_chart):
        c = base_chart.configure(
            axis_x=AxisConfig(label_angle=0),
            axis_y=AxisConfig(label_angle=-30),
        )
        cfg = c._configure[0]
        assert cfg.axis_x is not None
        assert cfg.axis_y is not None
        assert cfg.axis is None


# ---------------------------------------------------------------------------
# Chart.__add__ dispatch
# ---------------------------------------------------------------------------


class TestAddDispatch:
    def test_configure_dispatch(self, base_chart):
        cfg = Configure(axis=AxisConfig(label_angle=-45))
        c = base_chart + cfg
        assert len(c._configure) == 1
        assert c._configure[0] is cfg

    def test_configure_dispatch_does_not_mutate_original(self, base_chart):
        cfg = Configure(axis=AxisConfig(label_angle=-45))
        _ = base_chart + cfg
        assert base_chart._configure == []

    def test_annotate_container_dispatch(self, base_chart):
        import ferrum.annotation as ann

        container = Annotate([ann.text(1.0, 2.0, "label")])
        c = base_chart + container
        assert len(c._annotations) == 1
        assert c._annotations[0] is container

    def test_annotate_dispatch_does_not_mutate_original(self, base_chart):
        import ferrum.annotation as ann

        container = Annotate([ann.text(1.0, 2.0, "label")])
        _ = base_chart + container
        assert base_chart._annotations == []

    def test_bare_annotation_primitive_wraps_in_annotate(self, base_chart):
        import ferrum.annotation as ann

        primitive = ann.text(1.0, 2.0, "hi")
        c = base_chart + primitive
        assert len(c._annotations) == 1
        assert isinstance(c._annotations[0], Annotate)
        assert len(c._annotations[0].items) == 1
        assert isinstance(c._annotations[0].items[0], AnnotationText)

    def test_bare_primitive_does_not_mutate_original(self, base_chart):
        import ferrum.annotation as ann

        _ = base_chart + ann.text(1.0, 2.0, "hi")
        assert base_chart._annotations == []

    def test_secondary_y_dispatch(self, base_chart):
        sy = SecondaryY(field="y")
        c = base_chart + sy
        assert len(c._structural) == 1
        assert c._structural[0] is sy

    def test_break_axis_dispatch(self, base_chart):
        ba = BreakAxis(axis="y", gap=(50, 100))
        c = base_chart + ba
        assert len(c._structural) == 1
        assert c._structural[0] is ba

    def test_inset_dispatch(self, base_chart):
        df = pl.DataFrame({"x": [1, 2], "y": [3, 4]})
        inset_chart = fm.Chart(df).mark_point().encode(x="x", y="y")
        inset = Inset(chart=inset_chart, bounds=(0.5, 0.5, 1.0, 1.0))
        c = base_chart + inset
        assert len(c._structural) == 1
        assert c._structural[0] is inset

    def test_structural_does_not_mutate_original(self, base_chart):
        sy = SecondaryY(field="y")
        _ = base_chart + sy
        assert base_chart._structural == []

    def test_chart_plus_chart_still_works(self, base_chart):
        df = pl.DataFrame({"x": [1, 2, 3], "y": [7, 8, 9]})
        other = fm.Chart(df).mark_line().encode(x="x", y="y")
        layered = base_chart + other
        assert isinstance(layered, fm.Chart)
        assert layered._layers is not None
        assert len(layered._layers) == 2

    def test_unsupported_type_returns_not_implemented(self, base_chart):
        result = base_chart.__add__("not_a_chart")
        assert result is NotImplemented

    def test_multiple_structural_accumulate(self, base_chart):
        ba1 = BreakAxis(axis="y", gap=(10, 20))
        ba2 = BreakAxis(axis="x", gap=(5, 15))
        c = base_chart + ba1 + ba2
        assert len(c._structural) == 2

    def test_multiple_configure_layers_via_add(self, base_chart):
        c1 = Configure(axis=AxisConfig(label_angle=-45))
        c2 = Configure(legend=LegendConfig(orient="top"))
        c = base_chart + c1 + c2
        assert len(c._configure) == 2
