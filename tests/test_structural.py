"""Tests for ferrum.structural — SecondaryY, BreakAxis, and Inset."""

from __future__ import annotations

import pytest
import polars as pl

import ferrum as fm
from ferrum.structural import BreakAxis, Inset, SecondaryY


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def dual_field_df():
    return pl.DataFrame(
        {
            "x": [1, 2, 3, 4, 5],
            "y": [10, 20, 30, 40, 50],
            "revenue": [100, 200, 150, 300, 250],
        }
    )


@pytest.fixture()
def base_chart(dual_field_df):
    return fm.Chart(dual_field_df).mark_point().encode(x="x", y="y")


# ---------------------------------------------------------------------------
# SecondaryY
# ---------------------------------------------------------------------------


class TestSecondaryY:
    def test_default_construction(self):
        s = SecondaryY("revenue")
        assert s.field == "revenue"
        assert s.mark == "line"
        assert s.axis is None
        assert s.color is None
        assert s.opacity is None
        assert s.scale is None

    def test_frozen(self):
        s = SecondaryY("revenue")
        with pytest.raises((TypeError, AttributeError)):
            s.mark = "bar"  # type: ignore[misc]

    def test_field_stored(self):
        s = SecondaryY("profit")
        assert s.field == "profit"

    def test_mark_override(self):
        s = SecondaryY("revenue", mark="bar")
        assert s.mark == "bar"

    def test_color_set(self):
        s = SecondaryY("revenue", color="#e45756")
        assert s.color == "#e45756"

    def test_opacity_set(self):
        s = SecondaryY("revenue", opacity=0.7)
        assert s.opacity == 0.7

    def test_axis_set(self):
        # axis accepts any value (Axis instance or None)
        s = SecondaryY("revenue", axis=object())
        assert s.axis is not None

    def test_scale_set(self):
        # scale accepts any value (Scale instance or None)
        s = SecondaryY("revenue", scale=object())
        assert s.scale is not None

    def test_all_fields(self):
        sentinel_axis = object()
        sentinel_scale = object()
        s = SecondaryY(
            "revenue",
            mark="bar",
            axis=sentinel_axis,
            color="steelblue",
            opacity=0.5,
            scale=sentinel_scale,
        )
        assert s.field == "revenue"
        assert s.mark == "bar"
        assert s.axis is sentinel_axis
        assert s.color == "steelblue"
        assert s.opacity == 0.5
        assert s.scale is sentinel_scale


# ---------------------------------------------------------------------------
# BreakAxis
# ---------------------------------------------------------------------------


class TestBreakAxis:
    def test_valid_construction_y(self):
        b = BreakAxis(axis="y", gap=(50, 200))
        assert b.axis == "y"
        assert b.gap == (50, 200)
        assert b.break_size == 12
        assert b.break_style == "slash"

    def test_valid_construction_x(self):
        b = BreakAxis(axis="x", gap=(0, 1))
        assert b.axis == "x"

    def test_frozen(self):
        b = BreakAxis(axis="y", gap=(0, 1))
        with pytest.raises((TypeError, AttributeError)):
            b.axis = "x"  # type: ignore[misc]

    def test_invalid_axis_raises(self):
        with pytest.raises(ValueError, match="'x' or 'y'"):
            BreakAxis(axis="z", gap=(0, 1))

    def test_invalid_break_style_raises(self):
        with pytest.raises(ValueError, match="break_style"):
            BreakAxis(axis="x", gap=(0, 1), break_style="invalid")

    @pytest.mark.parametrize("style", ["slash", "zigzag", "wave", "gap"])
    def test_valid_break_styles(self, style):
        b = BreakAxis(axis="y", gap=(10, 90), break_style=style)
        assert b.break_style == style

    def test_custom_break_size(self):
        b = BreakAxis(axis="x", gap=(5, 50), break_size=20)
        assert b.break_size == 20

    def test_multiple_gaps_as_list(self):
        b = BreakAxis(axis="y", gap=[(10, 20), (50, 60)])
        assert b.gap == [(10, 20), (50, 60)]


# ---------------------------------------------------------------------------
# Inset
# ---------------------------------------------------------------------------


class TestInset:
    def test_default_construction(self):
        sentinel_chart = object()
        inset = Inset(chart=sentinel_chart, bounds=(0.1, 0.1, 0.9, 0.9))
        assert inset.chart is sentinel_chart
        assert inset.bounds == (0.1, 0.1, 0.9, 0.9)
        assert inset.border is True
        assert inset.border_color == "#999"
        assert inset.border_dash is None
        assert inset.background == "#fff"
        assert inset.shadow is False
        assert inset.connect_to is None
        assert inset.connect_style == "lines"

    def test_frozen(self):
        inset = Inset(chart=object(), bounds=(0, 0, 1, 1))
        with pytest.raises((TypeError, AttributeError)):
            inset.border = False  # type: ignore[misc]

    def test_invalid_connect_style_raises(self):
        with pytest.raises(ValueError, match="connect_style"):
            Inset(chart=object(), bounds=(0, 0, 1, 1), connect_style="arrow")

    @pytest.mark.parametrize("style", ["bracket", "lines", "none"])
    def test_valid_connect_styles(self, style):
        inset = Inset(chart=object(), bounds=(0, 0, 1, 1), connect_style=style)
        assert inset.connect_style == style

    def test_border_false(self):
        inset = Inset(chart=object(), bounds=(0, 0, 1, 1), border=False)
        assert inset.border is False

    def test_shadow_enabled(self):
        inset = Inset(chart=object(), bounds=(0, 0, 1, 1), shadow=True)
        assert inset.shadow is True

    def test_background_none(self):
        inset = Inset(chart=object(), bounds=(0, 0, 1, 1), background=None)
        assert inset.background is None

    def test_border_dash_set(self):
        inset = Inset(chart=object(), bounds=(0, 0, 1, 1), border_dash=[4, 4])
        assert inset.border_dash == [4, 4]

    def test_connect_to_set(self):
        inset = Inset(chart=object(), bounds=(0, 0, 1, 1), connect_to=(0.5, 0.5))
        assert inset.connect_to == (0.5, 0.5)


# ---------------------------------------------------------------------------
# Serialization tests (_resolve_chart_config)
# ---------------------------------------------------------------------------


class TestStructuralSerialization:
    def test_secondary_y_serialization(self, base_chart):
        chart = base_chart + SecondaryY("revenue")
        cfg = chart._resolve_chart_config()
        assert "structural" in cfg
        assert len(cfg["structural"]) == 1
        s = cfg["structural"][0]
        assert s["type"] == "secondary_y"
        assert s["field"] == "revenue"
        assert s["mark"] == "line"
        # Optional fields omitted when None
        assert "color" not in s
        assert "opacity" not in s

    def test_secondary_y_with_optional_fields(self, base_chart):
        chart = base_chart + SecondaryY("revenue", mark="bar", color="#e45756", opacity=0.7)
        cfg = chart._resolve_chart_config()
        s = cfg["structural"][0]
        assert s["mark"] == "bar"
        assert s["color"] == "#e45756"
        assert s["opacity"] == 0.7

    def test_break_axis_single_gap_serialization(self, base_chart):
        chart = base_chart + BreakAxis(axis="y", gap=(50, 200))
        cfg = chart._resolve_chart_config()
        assert "structural" in cfg
        s = cfg["structural"][0]
        assert s["type"] == "break_axis"
        assert s["axis"] == "y"
        # Single tuple normalized to nested list
        assert s["gaps"] == [[50, 200]]
        assert s["break_size"] == 12
        assert s["break_style"] == "slash"

    def test_break_axis_multiple_gaps(self, base_chart):
        chart = base_chart + BreakAxis(axis="y", gap=[(10, 20), (50, 60)])
        cfg = chart._resolve_chart_config()
        s = cfg["structural"][0]
        assert s["gaps"] == [[10, 20], [50, 60]]

    def test_break_axis_custom_style(self, base_chart):
        chart = base_chart + BreakAxis(axis="x", gap=(0, 5), break_size=20, break_style="zigzag")
        cfg = chart._resolve_chart_config()
        s = cfg["structural"][0]
        assert s["axis"] == "x"
        assert s["break_size"] == 20
        assert s["break_style"] == "zigzag"

    def test_inset_serialization(self, base_chart, dual_field_df):
        inset_chart = fm.Chart(dual_field_df).mark_point().encode(x="x", y="revenue")
        chart = base_chart + Inset(chart=inset_chart, bounds=(0.6, 0.1, 0.95, 0.45))
        cfg = chart._resolve_chart_config()
        assert "structural" in cfg
        s = cfg["structural"][0]
        assert s["type"] == "inset"
        assert isinstance(s["svg"], str)
        assert s["svg"].startswith("<svg")
        assert s["bounds"] == [0.6, 0.1, 0.95, 0.45]
        assert s["border"] is True
        assert s["border_color"] == "#999"
        assert s["background"] == "#fff"
        assert s["shadow"] is False
        assert s["connect_style"] == "lines"
        # Optional fields omitted when None
        assert "border_dash" not in s
        assert "connect_to" not in s

    def test_inset_optional_fields(self, base_chart, dual_field_df):
        inset_chart = fm.Chart(dual_field_df).mark_point().encode(x="x", y="revenue")
        chart = base_chart + Inset(
            chart=inset_chart,
            bounds=(0.5, 0.5, 1.0, 1.0),
            border_dash=[4, 4],
            connect_to=(2.0, 20.0),
            shadow=True,
            background=None,
        )
        cfg = chart._resolve_chart_config()
        s = cfg["structural"][0]
        assert s["border_dash"] == [4, 4]
        assert s["connect_to"] == [2.0, 20.0]
        assert s["shadow"] is True
        assert s["background"] is None

    def test_multiple_structural_features(self, base_chart):
        chart = base_chart + SecondaryY("revenue") + BreakAxis(axis="y", gap=(10, 90))
        cfg = chart._resolve_chart_config()
        assert "structural" in cfg
        assert len(cfg["structural"]) == 2
        types = {s["type"] for s in cfg["structural"]}
        assert types == {"secondary_y", "break_axis"}

    def test_no_structural_key_when_empty(self, base_chart):
        cfg = base_chart._resolve_chart_config()
        assert "structural" not in cfg

    def test_structural_with_configure(self, base_chart):
        from ferrum.configure import Configure, AxisConfig

        chart = base_chart + Configure(axis=AxisConfig(grid=True)) + SecondaryY("revenue")
        cfg = chart._resolve_chart_config()
        assert "structural" in cfg
        assert cfg["structural"][0]["type"] == "secondary_y"
        # Configure entry is also present
        assert "axis" in cfg


# ---------------------------------------------------------------------------
# Rendering integration tests (show_svg round-trip)
# ---------------------------------------------------------------------------


class TestStructuralRendering:
    def test_secondary_y_renders_svg(self, base_chart):
        chart = base_chart + SecondaryY("revenue")
        svg = chart.show_svg()
        assert svg.startswith("<svg")

    def test_break_axis_renders_svg(self, base_chart):
        chart = base_chart + BreakAxis(axis="y", gap=(50, 200))
        svg = chart.show_svg()
        assert svg.startswith("<svg")

    def test_inset_renders_svg(self, base_chart, dual_field_df):
        inset_chart = fm.Chart(dual_field_df).mark_point().encode(x="x", y="revenue")
        chart = base_chart + Inset(chart=inset_chart, bounds=(0.6, 0.1, 0.95, 0.45))
        svg = chart.show_svg()
        assert svg.startswith("<svg")

    def test_structural_with_configure_renders(self, base_chart):
        from ferrum.configure import Configure, AxisConfig

        chart = base_chart + Configure(axis=AxisConfig(grid=True)) + SecondaryY("revenue")
        svg = chart.show_svg()
        assert svg.startswith("<svg")
