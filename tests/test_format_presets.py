"""Tests for ferrum.format_presets — named format preset registry."""

from __future__ import annotations

import pytest

from ferrum.format_presets import (
    ALL_PRESETS,
    NUMERIC_PRESETS,
    TIME_PRESETS,
    is_time_preset,
    resolve_format,
)


class TestResolveFormat:
    @pytest.mark.parametrize(
        "name,expected",
        [
            ("integer", ",.0f"),
            ("decimal", ",.2f"),
            ("decimal1", ",.1f"),
            ("percent", ".1%"),
            ("percent_int", ".0%"),
            ("si", ".2s"),
            ("currency", "$,.0f"),
            ("currency_cents", "$,.2f"),
            ("compact", ".2~s"),
            ("scientific", ".2e"),
            ("ordinal", "__ordinal__"),
        ],
    )
    def test_numeric_presets(self, name, expected):
        assert resolve_format(name) == expected

    @pytest.mark.parametrize(
        "name,expected",
        [
            ("date_short", "%b %-d"),
            ("date_long", "%B %-d, %Y"),
            ("date_iso", "%Y-%m-%d"),
            ("month", "%b"),
            ("month_year", "%b %Y"),
            ("year", "%Y"),
            ("time", "%H:%M"),
            ("time_12h", "%-I:%M %p"),
            ("datetime", "%b %-d, %H:%M"),
        ],
    )
    def test_time_presets(self, name, expected):
        assert resolve_format(name) == expected

    def test_unknown_preset_raises_value_error(self):
        with pytest.raises(ValueError, match="Unknown format preset"):
            resolve_format("not_a_real_preset")

    def test_error_message_includes_preset_name(self):
        with pytest.raises(ValueError, match="not_a_real_preset"):
            resolve_format("not_a_real_preset")

    def test_all_presets_covered(self):
        """Every key in ALL_PRESETS must resolve without error."""
        for name in ALL_PRESETS:
            result = resolve_format(name)
            assert isinstance(result, str)
            assert len(result) > 0


class TestIsTimePreset:
    @pytest.mark.parametrize(
        "name",
        [
            "date_short",
            "date_long",
            "date_iso",
            "month",
            "month_year",
            "year",
            "time",
            "time_12h",
            "datetime",
        ],
    )
    def test_time_presets_return_true(self, name):
        assert is_time_preset(name) is True

    @pytest.mark.parametrize(
        "name",
        [
            "integer",
            "decimal",
            "decimal1",
            "percent",
            "percent_int",
            "si",
            "currency",
            "currency_cents",
            "compact",
            "scientific",
            "ordinal",
        ],
    )
    def test_numeric_presets_return_false(self, name):
        assert is_time_preset(name) is False

    def test_unknown_name_returns_false(self):
        assert is_time_preset("not_a_preset") is False

    def test_numeric_and_time_are_disjoint(self):
        numeric_keys = set(NUMERIC_PRESETS)
        time_keys = set(TIME_PRESETS)
        assert numeric_keys.isdisjoint(time_keys)

    def test_all_presets_equals_union(self):
        assert set(ALL_PRESETS) == set(NUMERIC_PRESETS) | set(TIME_PRESETS)
