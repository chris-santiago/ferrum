"""Tests for ferrum.configure — config dataclasses and the Configure container."""

from __future__ import annotations

import warnings

import pytest

from ferrum.configure import (
    AxisConfig,
    ColorConfig,
    Configure,
    GridConfig,
    LegendConfig,
    PaddingConfig,
    TitleConfig,
)


# ---------------------------------------------------------------------------
# AxisConfig
# ---------------------------------------------------------------------------


class TestAxisConfig:
    def test_default_construction(self):
        cfg = AxisConfig()
        assert cfg.x is True
        assert cfg.y is True
        assert cfg.label_angle is None

    def test_frozen(self):
        cfg = AxisConfig(label_angle=45)
        with pytest.raises((TypeError, AttributeError)):
            cfg.label_angle = 0  # type: ignore[misc]

    def test_to_dict_omits_none(self):
        cfg = AxisConfig(label_angle=-30, label_font_size=10)
        d = cfg.to_dict()
        assert d["label_angle"] == -30
        assert d["label_font_size"] == 10
        assert "label_color" not in d

    def test_to_dict_includes_non_none_booleans(self):
        cfg = AxisConfig(x=True, y=False, grid=False)
        d = cfg.to_dict()
        assert d["grid"] is False

    def test_to_dict_omits_deprecated_xy_keys(self):
        """x/y are vestigial no-ops (BUG 3); the wire schema does not accept
        them, so to_dict() must never emit either key (NF-B1 gate prep)."""
        cfg = AxisConfig(x=True, y=False, grid=False)
        d = cfg.to_dict()
        assert "x" not in d
        assert "y" not in d

    def test_label_format_and_raw_mutually_exclusive(self):
        with pytest.raises(ValueError, match="mutually exclusive"):
            AxisConfig(label_format="percent", label_format_raw=".1%")

    def test_label_format_alone_is_valid(self):
        cfg = AxisConfig(label_format="percent")
        assert cfg.label_format == "percent"
        assert cfg.label_format_raw is None

    def test_label_format_raw_alone_is_valid(self):
        cfg = AxisConfig(label_format_raw=".2f")
        assert cfg.label_format_raw == ".2f"
        assert cfg.label_format is None

    def test_full_round_trip(self):
        cfg = AxisConfig(
            label_angle=-45,
            tick_count=5,
            grid_color="#eee",
            domain_min=0.0,
            domain_max=100.0,
            nice=True,
            zero=False,
        )
        d = cfg.to_dict()
        assert d["label_angle"] == -45
        assert d["tick_count"] == 5
        assert d["grid_color"] == "#eee"
        assert d["domain_min"] == 0.0
        assert d["domain_max"] == 100.0
        assert d["nice"] is True
        assert d["zero"] is False

    def test_orphan_fields_default_none(self):
        cfg = AxisConfig()
        assert cfg.grid_opacity is None
        assert cfg.orient is None
        assert cfg.translate is None
        assert cfg.min_band is None
        assert cfg.max_band is None
        assert cfg.tick_extra is None
        assert cfg.tick_min_step is None
        assert cfg.title_orient is None
        assert cfg.zindex is None

    def test_orphan_fields_to_dict(self):
        cfg = AxisConfig(
            grid_opacity=0.3,
            orient="top",
            translate=5.0,
            min_band=10.0,
            max_band=40.0,
            tick_extra=True,
            tick_min_step=2.0,
            title_orient="left",
            zindex=1,
        )
        d = cfg.to_dict()
        assert d["grid_opacity"] == 0.3
        assert d["orient"] == "top"
        assert d["translate"] == 5.0
        assert d["min_band"] == 10.0
        assert d["max_band"] == 40.0
        assert d["tick_extra"] is True
        assert d["tick_min_step"] == 2.0
        assert d["title_orient"] == "left"
        assert d["zindex"] == 1

    def test_orphan_fields_omitted_when_none(self):
        d = AxisConfig(label_angle=-30).to_dict()
        for key in (
            "grid_opacity",
            "orient",
            "translate",
            "min_band",
            "max_band",
            "tick_extra",
            "tick_min_step",
            "title_orient",
            "zindex",
        ):
            assert key not in d


# ---------------------------------------------------------------------------
# AxisConfig.x / .y deprecation (BUG 3 — vestigial flags, steer to Chart.axis)
# ---------------------------------------------------------------------------


class TestAxisXYDeprecation:
    """The vestigial ``x``/``y`` flags are silent no-ops.

    Setting either to ``False`` must emit a loud ``DeprecationWarning`` that
    points the user at the working ``Chart.axis(x=False)`` / ``Chart.axis(y=False)``
    API. The ``True`` defaults stay a silent no-op (no warning).
    """

    def test_axis_config_x_false_warns(self):
        with pytest.warns(DeprecationWarning, match="Chart.axis"):
            AxisConfig(x=False)

    def test_axis_config_y_false_warns(self):
        with pytest.warns(DeprecationWarning, match="Chart.axis"):
            AxisConfig(y=False)

    def test_axis_config_x_false_still_serializes(self):
        """The warning still fires, but to_dict() never emits the dead 'x' key
        (NF-B1, 2026-09-02): only the sibling field flows through."""
        with pytest.warns(DeprecationWarning):
            cfg = AxisConfig(x=False, label_angle=-45)
        d = cfg.to_dict()
        assert "x" not in d
        assert d["label_angle"] == -45

    def test_axis_config_defaults_do_not_warn(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            AxisConfig()
            AxisConfig(x=True, y=True)
            AxisConfig(label_angle=45)


# ---------------------------------------------------------------------------
# LegendConfig
# ---------------------------------------------------------------------------


class TestLegendConfig:
    def test_default_construction(self):
        cfg = LegendConfig()
        assert cfg.orient is None
        assert cfg.columns is None

    def test_frozen(self):
        cfg = LegendConfig(orient="top")
        with pytest.raises((TypeError, AttributeError)):
            cfg.orient = "left"  # type: ignore[misc]

    def test_to_dict_omits_none(self):
        cfg = LegendConfig(orient="top", columns=2)
        d = cfg.to_dict()
        assert d["orient"] == "top"
        assert d["columns"] == 2
        assert "label_font_size" not in d

    @pytest.mark.parametrize("orient", ["right", "left", "top", "bottom", "none"])
    def test_valid_orients(self, orient):
        cfg = LegendConfig(orient=orient)
        assert cfg.orient == orient

    def test_invalid_orient_raises(self):
        with pytest.raises(ValueError, match="orient"):
            LegendConfig(orient="center")

    def test_orient_none_is_valid(self):
        # orient=None means "not set" — different from orient="none"
        cfg = LegendConfig(orient=None)
        assert cfg.orient is None

    def test_styling_fields_default_none(self):
        cfg = LegendConfig()
        assert cfg.label_color is None
        assert cfg.label_limit is None
        assert cfg.symbol_stroke_width is None
        assert cfg.gradient_thickness is None
        assert cfg.title_padding is None
        assert cfg.row_padding is None
        assert cfg.column_padding is None
        assert cfg.clip_height is None
        assert cfg.tick_min_step is None
        assert cfg.zindex is None

    def test_styling_fields_to_dict(self):
        cfg = LegendConfig(
            label_color="#333333",
            label_limit=40.0,
            symbol_stroke_width=3.0,
            gradient_thickness=30.0,
            title_padding=15.0,
            row_padding=20.0,
            column_padding=30.0,
            clip_height=40.0,
            tick_min_step=5.0,
            zindex=1,
        )
        d = cfg.to_dict()
        assert d["label_color"] == "#333333"
        assert d["label_limit"] == 40.0
        assert d["symbol_stroke_width"] == 3.0
        assert d["gradient_thickness"] == 30.0
        assert d["title_padding"] == 15.0
        assert d["row_padding"] == 20.0
        assert d["column_padding"] == 30.0
        assert d["clip_height"] == 40.0
        assert d["tick_min_step"] == 5.0
        assert d["zindex"] == 1

    def test_styling_fields_omitted_when_none(self):
        d = LegendConfig(orient="top").to_dict()
        for key in (
            "label_color",
            "label_limit",
            "symbol_stroke_width",
            "gradient_thickness",
            "title_padding",
            "row_padding",
            "column_padding",
            "clip_height",
            "tick_min_step",
            "zindex",
        ):
            assert key not in d


# ---------------------------------------------------------------------------
# TitleConfig
# ---------------------------------------------------------------------------


class TestTitleConfig:
    def test_default_construction(self):
        cfg = TitleConfig()
        assert cfg.anchor is None
        assert cfg.font_size is None

    def test_frozen(self):
        cfg = TitleConfig(anchor="middle")
        with pytest.raises((TypeError, AttributeError)):
            cfg.anchor = "end"  # type: ignore[misc]

    def test_to_dict_omits_none(self):
        cfg = TitleConfig(font_size=18, anchor="start")
        d = cfg.to_dict()
        assert d["font_size"] == 18
        assert d["anchor"] == "start"
        assert "color" not in d

    @pytest.mark.parametrize("anchor", ["start", "middle", "end"])
    def test_valid_anchors(self, anchor):
        cfg = TitleConfig(anchor=anchor)
        assert cfg.anchor == anchor

    def test_invalid_anchor_raises(self):
        with pytest.raises(ValueError, match="anchor"):
            TitleConfig(anchor="left")

    def test_anchor_none_is_valid(self):
        cfg = TitleConfig(anchor=None)
        assert cfg.anchor is None


# ---------------------------------------------------------------------------
# GridConfig
# ---------------------------------------------------------------------------


class TestGridConfig:
    def test_default_construction(self):
        cfg = GridConfig()
        assert cfg.x is None
        assert cfg.color is None

    def test_frozen(self):
        cfg = GridConfig(color="#eee")
        with pytest.raises((TypeError, AttributeError)):
            cfg.color = "#fff"  # type: ignore[misc]

    def test_to_dict_omits_none(self):
        cfg = GridConfig(x=True, color="#ddd", width=0.5)
        d = cfg.to_dict()
        assert d["x"] is True
        assert d["color"] == "#ddd"
        assert d["width"] == 0.5
        assert "y" not in d
        assert "dash" not in d


# ---------------------------------------------------------------------------
# PaddingConfig
# ---------------------------------------------------------------------------


class TestPaddingConfig:
    """Class-shape tests only.

    Construction-time validation (negative/non-numeric/non-finite padding)
    is feature coverage for this task (NF-B5/B6/B7) and lives in
    ``tests/test_padding_validation.py`` per this project's test-file
    convention (findings-scoped/feature files own their feature's
    coverage); duplicating it here drifted the two files out of sync
    across review cycles.
    """

    def test_default_construction(self):
        cfg = PaddingConfig()
        assert cfg.auto is True
        assert cfg.top is None

    def test_frozen(self):
        cfg = PaddingConfig(top=10)
        with pytest.raises((TypeError, AttributeError)):
            cfg.top = 20  # type: ignore[misc]

    def test_to_dict_omits_none(self):
        cfg = PaddingConfig(top=10, left=5, auto=False)
        d = cfg.to_dict()
        assert d["top"] == 10
        assert d["left"] == 5
        assert d["auto"] is False
        assert "right" not in d
        assert "bottom" not in d


# ---------------------------------------------------------------------------
# ColorConfig
# ---------------------------------------------------------------------------


class TestColorConfig:
    def test_default_construction(self):
        cfg = ColorConfig()
        assert cfg.scheme is None
        assert cfg.domain is None

    def test_frozen(self):
        cfg = ColorConfig(scheme="tableau10")
        with pytest.raises((TypeError, AttributeError)):
            cfg.scheme = "viridis"  # type: ignore[misc]

    def test_to_dict_omits_none(self):
        cfg = ColorConfig(scheme="tableau10", domain=["a", "b"])
        d = cfg.to_dict()
        assert d["scheme"] == "tableau10"
        assert d["domain"] == ["a", "b"]
        assert "sequential_scheme" not in d


# ---------------------------------------------------------------------------
# Configure container
# ---------------------------------------------------------------------------


class TestConfigure:
    def test_empty_construction(self):
        cfg = Configure()
        assert cfg.axis is None
        assert cfg.legend is None

    def test_frozen(self):
        cfg = Configure(axis=AxisConfig(label_angle=45))
        with pytest.raises((TypeError, AttributeError)):
            cfg.axis = None  # type: ignore[misc]

    def test_to_dict_omits_none_fields(self):
        cfg = Configure(
            axis=AxisConfig(label_angle=30),
            legend=LegendConfig(orient="top"),
        )
        d = cfg.to_dict()
        assert "axis" in d
        assert "legend" in d
        assert "title" not in d
        assert "grid" not in d

    def test_to_dict_recurses_into_sub_configs(self):
        cfg = Configure(
            axis=AxisConfig(label_angle=30, grid_color="#eee"),
            title=TitleConfig(font_size=16),
        )
        d = cfg.to_dict()
        assert d["axis"]["label_angle"] == 30
        assert d["axis"]["grid_color"] == "#eee"
        assert d["title"]["font_size"] == 16

    def test_accepts_axis_x_y_y2(self):
        cfg = Configure(
            axis_x=AxisConfig(label_angle=0),
            axis_y=AxisConfig(label_angle=-30),
            axis_y2=AxisConfig(tick_count=5),
        )
        d = cfg.to_dict()
        assert "axis_x" in d
        assert "axis_y" in d
        assert "axis_y2" in d

    def test_accepts_all_config_types(self):
        cfg = Configure(
            axis=AxisConfig(),
            axis_x=AxisConfig(),
            axis_y=AxisConfig(),
            axis_y2=AxisConfig(),
            legend=LegendConfig(),
            title=TitleConfig(),
            grid=GridConfig(),
            padding=PaddingConfig(),
            color=ColorConfig(),
        )
        # All set, no error raised
        assert cfg.axis is not None
        assert cfg.color is not None
