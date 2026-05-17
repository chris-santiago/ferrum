"""Tests for Phase 12 Axis and Legend value classes."""

from __future__ import annotations

import pytest

import ferrum as fm
from ferrum.axis import Axis, _normalize_axis, _axis_suppressed_dict
from ferrum.legend import Legend, _normalize_legend


# ---------------------------------------------------------------------------
# Axis tests
# ---------------------------------------------------------------------------


class TestAxisFrozen:
    """Axis instances are immutable (frozen dataclass)."""

    def test_frozen(self):
        ax = Axis(title="Speed")
        with pytest.raises(AttributeError):
            ax.title = "Other"  # type: ignore[misc]

    def test_slots(self):
        ax = Axis()
        with pytest.raises((AttributeError, TypeError)):
            ax.new_attr = 42  # type: ignore[attr-defined]


class TestAxisToDict:
    """Axis.to_dict() serializes correctly, omitting defaults and None."""

    def test_empty(self):
        # All defaults — empty dict
        assert Axis().to_dict() == {}

    def test_title_and_grid_false(self):
        result = Axis(title="Speed", grid=False).to_dict()
        assert result == {"title": "Speed", "grid": False}

    def test_all_bool_suppressed(self):
        result = Axis(ticks=False, labels=False, domain=False, grid=False).to_dict()
        assert result == {"ticks": False, "labels": False, "domain": False, "grid": False}

    def test_tick_count(self):
        result = Axis(tick_count=5).to_dict()
        assert result == {"tick_count": 5}

    def test_grid_styling(self):
        result = Axis(grid_dash=[4.0, 2.0], grid_width=0.5, grid_color="#ccc").to_dict()
        assert result == {"grid_dash": [4.0, 2.0], "grid_width": 0.5, "grid_color": "#ccc"}

    def test_label_options(self):
        result = Axis(label_angle=45.0, label_overlap="greedy").to_dict()
        assert result == {"label_angle": 45.0, "label_overlap": "greedy"}

    def test_orient(self):
        result = Axis(orient="top").to_dict()
        assert result == {"orient": "top"}

    def test_values_explicit(self):
        result = Axis(values=[0, 25, 50, 75, 100]).to_dict()
        assert result == {"values": [0, 25, 50, 75, 100]}


class TestAxisNormalize:
    """_normalize_axis handles all input forms."""

    def test_none_passthrough(self):
        assert _normalize_axis(None) is None

    def test_false_suppression(self):
        result = _normalize_axis(False)
        assert result == _axis_suppressed_dict()
        assert result["ticks"] is False
        assert result["labels"] is False
        assert result["domain"] is False
        assert result["grid"] is False

    def test_axis_instance(self):
        ax = Axis(title="X", grid=False)
        result = _normalize_axis(ax)
        assert result == {"title": "X", "grid": False}

    def test_dict_passthrough(self):
        d = {"title": "Custom", "ticks": False}
        assert _normalize_axis(d) is d


# ---------------------------------------------------------------------------
# Legend tests
# ---------------------------------------------------------------------------


class TestLegendFrozen:
    """Legend instances are immutable (frozen dataclass)."""

    def test_frozen(self):
        lg = Legend(title="Species")
        with pytest.raises(AttributeError):
            lg.title = "Other"  # type: ignore[misc]

    def test_slots(self):
        lg = Legend()
        with pytest.raises((AttributeError, TypeError)):
            lg.new_attr = 42  # type: ignore[attr-defined]


class TestLegendToDict:
    """Legend.to_dict() serializes correctly, omitting defaults and None."""

    def test_empty(self):
        # orient="right" and direction="vertical" are defaults — omitted
        assert Legend().to_dict() == {}

    def test_orient_bottom_columns(self):
        result = Legend(orient="bottom", columns=3).to_dict()
        assert result == {"orient": "bottom", "columns": 3}

    def test_direction_horizontal(self):
        result = Legend(direction="horizontal").to_dict()
        assert result == {"direction": "horizontal"}

    def test_symbol_options(self):
        result = Legend(symbol_size=100.0, symbol_type="square").to_dict()
        assert result == {"symbol_size": 100.0, "symbol_type": "square"}

    def test_gradient_options(self):
        result = Legend(type="gradient", gradient_length=200.0).to_dict()
        assert result == {"type": "gradient", "gradient_length": 200.0}

    def test_title_and_font(self):
        result = Legend(title="Category", title_font_size=14.0, label_font_size=11.0).to_dict()
        assert result == {
            "title": "Category",
            "title_font_size": 14.0,
            "label_font_size": 11.0,
        }


class TestLegendNormalize:
    """_normalize_legend handles all input forms."""

    def test_none_suppression(self):
        result = _normalize_legend(None)
        assert result == {"disabled": True}

    def test_false_suppression(self):
        result = _normalize_legend(False)
        assert result == {"disabled": True}

    def test_legend_instance(self):
        lg = Legend(orient="bottom", columns=2)
        result = _normalize_legend(lg)
        assert result == {"orient": "bottom", "columns": 2}

    def test_dict_passthrough(self):
        d = {"orient": "left", "title": "Size"}
        assert _normalize_legend(d) is d

    def test_other_truthy_reserved(self):
        # Unknown truthy values -> None (reserved)
        assert _normalize_legend(True) is None
        assert _normalize_legend(42) is None


# ---------------------------------------------------------------------------
# Integration with encoding channels
# ---------------------------------------------------------------------------


class TestAxisEncodingIntegration:
    """Axis value class integrates with encoding channels."""

    def test_axis_false_on_x(self):
        enc = fm.X("field", axis=False)
        d = enc.to_encoding_spec_dict()
        assert d["axis"] == _axis_suppressed_dict()

    def test_axis_instance_on_x(self):
        enc = fm.X("field", axis=Axis(title="X Axis", grid=False))
        d = enc.to_encoding_spec_dict()
        assert d["axis"] == {"title": "X Axis", "grid": False}

    def test_axis_dict_on_x(self):
        """Backward compat: axis as raw dict still works."""
        enc = fm.X("field", axis={"title": "X"})
        d = enc.to_encoding_spec_dict()
        assert d["axis"] == {"title": "X"}

    def test_axis_instance_on_y(self):
        enc = fm.Y("mpg", axis=Axis(orient="right", tick_count=10))
        d = enc.to_encoding_spec_dict()
        assert d["axis"] == {"orient": "right", "tick_count": 10}

    def test_axis_none_not_set(self):
        """axis=None means 'not specified' — no axis key in output."""
        enc = fm.X("field")
        d = enc.to_encoding_spec_dict()
        assert "axis" not in d


class TestLegendEncodingIntegration:
    """Legend value class integrates with encoding channels."""

    def test_legend_none_suppresses(self):
        enc = fm.Color("species", legend=None)
        d = enc.to_encoding_spec_dict()
        assert d["legend"] == {"disabled": True}

    def test_legend_false_suppresses(self):
        enc = fm.Color("species", legend=False)
        d = enc.to_encoding_spec_dict()
        assert d["legend"] == {"disabled": True}

    def test_legend_instance_on_color(self):
        enc = fm.Color("species", legend=Legend(orient="bottom"))
        d = enc.to_encoding_spec_dict()
        assert d["legend"] == {"orient": "bottom"}

    def test_legend_dict_on_color(self):
        """Backward compat: legend as raw dict still works."""
        enc = fm.Color("species", legend={"orient": "left"})
        d = enc.to_encoding_spec_dict()
        assert d["legend"] == {"orient": "left"}

    def test_legend_instance_with_columns(self):
        enc = fm.Color("origin", legend=Legend(orient="bottom", columns=3, direction="horizontal"))
        d = enc.to_encoding_spec_dict()
        assert d["legend"] == {"orient": "bottom", "columns": 3, "direction": "horizontal"}


# ---------------------------------------------------------------------------
# Public API availability
# ---------------------------------------------------------------------------


class TestPublicExports:
    """Axis and Legend are importable from ferrum top level."""

    def test_axis_importable(self):
        assert fm.Axis is Axis

    def test_legend_importable(self):
        assert fm.Legend is Legend
