"""Tests for the GD6 conditional / ``fm.when`` channel-resolution fix.

Two linked correctness bugs in the D6 conditional surface:

D-1 (silent no-op): ``fm.when(sel).then(1.0).otherwise(0.15)`` — the
"fade unselected" idiom from ``fm.when``'s own docstring — used to serialize
a conditional with ``channel="color"`` but ``if_selected={"kind":"opacity"}``.
That mismatched ``(channel, kind)`` tuple hit no arm in the WASM matcher, so the
fade silently did nothing.

D-2 (TypeError): ``encode(color=sel.when(...).otherwise(...))`` raised
``TypeError`` even though several docstrings show this inline form.

The fix plumbs an explicit channel through ``ConditionalSpec`` (taken from the
``encode`` key) so both the wire ``channel`` and the value-kind resolution are
correct, and fixes ``_resolve_channel`` to default a bare number to
``"opacity"`` (the only numeric default with a matching WASM arm).
"""

from __future__ import annotations

import json

import ferrum as fm
import polars as pl
import pytest
from ferrum.selection import _resolve_channel, value


@pytest.fixture
def df() -> pl.DataFrame:
    return pl.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6], "c": ["a", "b", "a"]})


def _conditionals(chart: fm.Chart) -> list[dict]:
    spec = json.loads(chart.to_json())
    return spec.get("conditionals", [])


def _selection_names(chart: fm.Chart) -> set[str]:
    spec = json.loads(chart.to_json())
    return {s["name"] for s in spec.get("selections", [])}


# ── D-1: encode(opacity=fm.when(...).then(num).otherwise(num)) ───────────────


class TestEncodeOpacityConditional:
    def test_opacity_channel_and_kinds(self, df):
        sel = fm.selection_point()
        chart = (
            fm.Chart(df)
            .mark_point()
            .encode(
                x="x",
                y="y",
                opacity=fm.when(sel).then(1.0).otherwise(0.15),
            )
        )
        conds = _conditionals(chart)
        assert len(conds) == 1
        cond = conds[0]
        assert cond["channel"] == "opacity"
        assert cond["if_selected"] == {"kind": "opacity", "value": 1.0}
        assert cond["if_not"] == {"kind": "opacity", "value": 0.15}

    def test_selection_auto_registered(self, df):
        sel = fm.selection_point(name="mysel")
        chart = (
            fm.Chart(df)
            .mark_point()
            .encode(
                x="x",
                y="y",
                opacity=fm.when(sel).then(1.0).otherwise(0.15),
            )
        )
        # No separate .add_selection() call — the conditional auto-registers it.
        assert "mysel" in _selection_names(chart)

    def test_no_type_error(self, df):
        sel = fm.selection_point()
        # The whole point: this used to raise TypeError at encode.
        fm.Chart(df).mark_point().encode(
            x="x", y="y", opacity=fm.when(sel).then(1.0).otherwise(0.15)
        )


# ── D-1: encode(size=...) ─────────────────────────────────────────────────────


class TestEncodeSizeConditional:
    def test_size_channel_and_kinds(self, df):
        sel = fm.selection_point()
        chart = (
            fm.Chart(df)
            .mark_point()
            .encode(
                x="x",
                y="y",
                size=fm.when(sel).then(100).otherwise(20),
            )
        )
        conds = _conditionals(chart)
        assert len(conds) == 1
        cond = conds[0]
        assert cond["channel"] == "size"
        assert cond["if_selected"] == {"kind": "size", "value": 100.0}
        assert cond["if_not"] == {"kind": "size", "value": 20.0}


# ── D-2: encode(color=sel.when(Color(...)).otherwise(value("#ccc"))) ──────────


class TestEncodeColorConditional:
    def test_color_channel_and_field_kind(self, df):
        sel = fm.selection_point()
        chart = (
            fm.Chart(df)
            .mark_point()
            .encode(
                x="x",
                y="y",
                color=sel.when(fm.Color("c")).otherwise(value("#ccc")),
            )
        )
        conds = _conditionals(chart)
        assert len(conds) == 1
        cond = conds[0]
        assert cond["channel"] == "color"
        assert cond["if_selected"] == {"kind": "field", "name": "c"}

    def test_no_type_error(self, df):
        sel = fm.selection_point()
        fm.Chart(df).mark_point().encode(
            x="x", y="y", color=sel.when(fm.Color("c")).otherwise(value("#ccc"))
        )

    def test_selection_auto_registered(self, df):
        sel = fm.selection_point(name="csel")
        chart = (
            fm.Chart(df)
            .mark_point()
            .encode(
                x="x",
                y="y",
                color=sel.when(fm.Color("c")).otherwise(value("#ccc")),
            )
        )
        assert "csel" in _selection_names(chart)


# ── _resolve_channel: bare number defaults to opacity ─────────────────────────


class TestResolveChannelDefault:
    def test_bare_number_defaults_to_opacity(self):
        assert _resolve_channel(value(0.5)) == "opacity"

    def test_hex_string_resolves_to_color(self):
        assert _resolve_channel(value("#ccc")) == "color"


# ── Back-compat: explicit .add_selection(...).conditional(...) path ───────────


class TestBackCompatConditionalPath:
    def test_color_conditional_unchanged(self, df):
        sel = fm.selection_point(fields=["c"])
        chart = (
            fm.Chart(df)
            .mark_point()
            .encode(x="x", y="y")
            .add_selection(sel)
            .conditional(sel.when(fm.Color("c")).otherwise(value("#ccc")))
        )
        conds = _conditionals(chart)
        assert len(conds) == 1
        cond = conds[0]
        assert cond["channel"] == "color"
        assert cond["if_selected"] == {"kind": "field", "name": "c"}

    def test_conditional_auto_registers_selection(self, df):
        # The explicit path also benefits: .conditional() auto-registers the
        # carried selection when present, so no separate .add_selection().
        sel = fm.selection_point(name="autoreg")
        chart = (
            fm.Chart(df)
            .mark_point()
            .encode(x="x", y="y")
            .conditional(sel.when(fm.Color("c")).otherwise(value("#ccc")))
        )
        assert "autoreg" in _selection_names(chart)


# ── Non-conditional encode is unaffected ──────────────────────────────────────


class TestPlainEncodeUnaffected:
    def test_plain_encode_no_conditionals(self, df):
        chart = fm.Chart(df).mark_point().encode(x="x", y="y", color="c")
        assert _conditionals(chart) == []
