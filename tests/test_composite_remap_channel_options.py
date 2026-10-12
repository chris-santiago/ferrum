"""Pins that a composite mark's remapped x/y channel keeps every option the caller wrote.

``mark_histogram``, 1-D ``mark_density``, ``mark_smooth`` without ``ci``, and
``mark_function`` re-point the caller's ``x``/``y`` channel at a transform's
output column during desugar (``_apply_remap``, ``src/ferrum/_desugar.py``).
The remapped channel keeps every option the caller wrote on it (``title=``,
``Axis(title=)``, ``format=``, ``format_type=``, an explicit ``scale=``);
only the now-resolved ``bin=``/``aggregate=`` kwargs are dropped and the
type is forced to quantitative. A channel the caller never bound gets a
fresh quantitative channel, stamped with the original field name as its
title only when the remap changed the field and no explicit title was
given.

Every test here either exercises the public mark API end to end (so a
rendering-pipeline regression would show up too), or calls ``_apply_remap``
directly with real ``ChannelBase`` instances built the way ``Chart.encode()``
builds them -- the same function every caller above invokes, not a
hand-built stand-in for it.
"""

from __future__ import annotations

import re

import polars as pl
import pytest

import ferrum as fm
from ferrum._desugar import _apply_remap
from ferrum.encoding import X, X2, Y, Y2
from ferrum.repeat import Repeat


def _axis_title_texts(svg: str) -> list[str]:
    """Rendered axis-title text, in document order.

    Axis titles are the only ``<text>`` elements the renderer draws with
    ``font-weight="600"``; tick labels and other text carry no such weight.
    """
    return re.findall(r'font-weight="600"[^>]*>([^<]+)</text>', svg)


def _numeric_tick_texts(svg: str) -> list[str]:
    """Every rendered tick-label string that looks like a formatted number."""
    return re.findall(r"<text[^>]*>([0-9.]+\.[0-9]+)</text>", svg)


@pytest.fixture()
def hist_df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "measurement": [
                1.0,
                1.4,
                1.8,
                2.1,
                2.3,
                2.6,
                2.9,
                3.2,
                3.6,
                4.0,
                4.3,
                4.8,
                5.1,
                5.6,
                6.0,
            ]
            * 2
        }
    )


@pytest.fixture()
def smooth_df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "time": [float(v) for v in range(1, 11)] * 2,
            "temp": [2.0, 2.5, 3.0, 4.1, 4.8, 5.9, 6.7, 7.5, 8.2, 9.9] * 2,
            # Field names matching the Smooth transform's own output columns,
            # so a remap over this pair never renames the field.
            "x": [float(v) for v in range(1, 11)] * 2,
            "y": [2.0, 2.5, 3.0, 4.1, 4.8, 5.9, 6.7, 7.5, 8.2, 9.9] * 2,
        }
    )


# ---------------------------------------------------------------------------
# mark_histogram
# ---------------------------------------------------------------------------


class TestHistogramKeepsCallerOptions:
    def test_title_survives(self, hist_df):
        chart = fm.Chart(hist_df).mark_histogram().encode(x=fm.X("measurement", title="Value"))
        resolved = chart._resolve_pending()
        assert resolved._encoding["x"].option("title") == "Value"
        assert "Value" in _axis_title_texts(chart.to_svg())

    def test_horizontal_orientation_honors_both_sides(self, hist_df):
        chart = (
            fm.Chart(hist_df)
            .mark_histogram(orientation="horizontal")
            .encode(y=fm.Y("measurement", title="Value"), x=fm.X("count", title="Count"))
        )
        resolved = chart._resolve_pending()
        assert resolved._encoding["y"].option("title") == "Value"
        assert resolved._encoding["x"].option("title") == "Count"
        titles = _axis_title_texts(chart.to_svg())
        assert "Value" in titles
        assert "Count" in titles

    def test_axis_title_honored(self, hist_df):
        chart = (
            fm.Chart(hist_df)
            .mark_histogram()
            .encode(x=fm.X("measurement", axis=fm.Axis(title="AxT")))
        )
        resolved = chart._resolve_pending()
        assert resolved._encoding["x"].option("axis").title == "AxT"
        assert "AxT" in _axis_title_texts(chart.to_svg())

    def test_format_and_format_type_honored(self, hist_df):
        chart = fm.Chart(hist_df).mark_histogram().encode(x=fm.X("measurement", format=".2f"))
        resolved = chart._resolve_pending()
        assert resolved._encoding["x"].option("format") == ".2f"
        assert "1.00" in _numeric_tick_texts(chart.to_svg())

    def test_encode_before_and_after_mark_are_equivalent(self, hist_df):
        before = fm.Chart(hist_df).encode(x=fm.X("measurement", title="Value")).mark_histogram()
        after = fm.Chart(hist_df).mark_histogram().encode(x=fm.X("measurement", title="Value"))
        assert before._resolve_pending()._encoding["x"] == after._resolve_pending()._encoding["x"]

    def test_source_type_is_forced_quantitative(self, hist_df):
        chart = fm.Chart(hist_df).mark_histogram().encode(x=fm.X("measurement", type="O"))
        resolved = chart._resolve_pending()
        assert resolved._encoding["x"].option("type") == "Q"

    def test_unbound_y_gets_a_fresh_quantitative_channel(self, hist_df):
        chart = fm.Chart(hist_df).mark_histogram().encode(x="measurement")
        resolved = chart._resolve_pending()
        assert resolved._encoding["y"] == fm.Y("count", type="Q")

    def test_explicit_scale_is_carried(self, hist_df):
        scale = fm.LinearScale(domain=[0, 10])
        chart = fm.Chart(hist_df).mark_histogram().encode(x=fm.X("measurement", scale=scale))
        resolved = chart._resolve_pending()
        assert resolved._encoding["x"].option("scale") == scale

    def test_untitled_default_stamps_the_original_field_name(self, hist_df):
        """No explicit title means the stamped field name is the only title."""
        chart = fm.Chart(hist_df).mark_histogram().encode(x="measurement")
        resolved = chart._resolve_pending()
        assert resolved._encoding["x"].option("title") == "measurement"

    def test_repeat_placeholder_is_deferred_before_remap(self):
        """An unresolved Repeat placeholder is never remapped."""
        df = pl.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "b": [4.0, 3.0, 2.0, 1.0]})
        chart = fm.Chart(df).mark_histogram().encode(x=Repeat.column)
        resolved = chart._resolve_pending()
        assert resolved._encoding["x"].field is Repeat.column


# ---------------------------------------------------------------------------
# mark_density (1-D)
# ---------------------------------------------------------------------------


class TestDensityKeepsCallerOptions:
    def test_title_survives(self, hist_df):
        chart = fm.Chart(hist_df).mark_density().encode(x=fm.X("measurement", title="Value"))
        resolved = chart._resolve_pending()
        assert resolved._encoding["x"].option("title") == "Value"
        assert "Value" in _axis_title_texts(chart.to_svg())

    def test_horizontal_orientation_honors_the_value_side(self, hist_df):
        chart = (
            fm.Chart(hist_df)
            .mark_density(orientation="horizontal")
            .encode(y=fm.Y("measurement", title="Value"))
        )
        resolved = chart._resolve_pending()
        assert resolved._encoding["y"].option("title") == "Value"
        assert "Value" in _axis_title_texts(chart.to_svg())

    def test_format_honored(self, hist_df):
        chart = fm.Chart(hist_df).mark_density().encode(x=fm.X("measurement", format=".2f"))
        resolved = chart._resolve_pending()
        assert resolved._encoding["x"].option("format") == ".2f"

    def test_unbound_density_axis_gets_a_fresh_quantitative_channel(self, hist_df):
        chart = fm.Chart(hist_df).mark_density().encode(x="measurement")
        resolved = chart._resolve_pending()
        assert resolved._encoding["y"] == fm.Y("density", type="Q")

    def test_untitled_default_stamps_the_original_field_name(self, hist_df):
        """No explicit title means the stamped field name is the only title."""
        chart = fm.Chart(hist_df).mark_density().encode(x="measurement")
        resolved = chart._resolve_pending()
        assert resolved._encoding["x"].option("title") == "measurement"


# ---------------------------------------------------------------------------
# mark_smooth (no ci)
# ---------------------------------------------------------------------------


class TestSmoothNoCiKeepsCallerOptions:
    def test_title_survives(self, smooth_df):
        chart = (
            fm.Chart(smooth_df)
            .mark_smooth(method="lm")
            .encode(x=fm.X("time", title="Time"), y=fm.Y("temp", title="Temp"))
        )
        resolved = chart._resolve_pending()
        assert resolved._encoding["x"].option("title") == "Time"
        assert resolved._encoding["y"].option("title") == "Temp"
        titles = _axis_title_texts(chart.to_svg())
        assert "Time" in titles
        assert "Temp" in titles

    def test_format_honored(self, smooth_df):
        chart = (
            fm.Chart(smooth_df)
            .mark_smooth(method="lm")
            .encode(x=fm.X("time", format=".2f"), y="temp")
        )
        resolved = chart._resolve_pending()
        assert resolved._encoding["x"].option("format") == ".2f"

    def test_encode_before_and_after_mark_are_equivalent(self, smooth_df):
        enc = dict(x=fm.X("time", title="Time"), y=fm.Y("temp", title="Temp"))
        before = fm.Chart(smooth_df).encode(**enc).mark_smooth(method="lm")
        after = fm.Chart(smooth_df).mark_smooth(method="lm").encode(**enc)
        assert before._resolve_pending()._encoding["x"] == after._resolve_pending()._encoding["x"]
        assert before._resolve_pending()._encoding["y"] == after._resolve_pending()._encoding["y"]

    def test_untitled_default_keeps_the_field_name_when_unrenamed(self, smooth_df):
        """The smooth transform's own column names are "x"/"y": a chart
        already encoded on fields named "x"/"y" is never renamed, so no
        title stamp fires."""
        chart = fm.Chart(smooth_df).mark_smooth(method="lm").encode(x="x", y="y")
        resolved = chart._resolve_pending()
        assert resolved._encoding["x"].option("title") is None
        assert resolved._encoding["y"].option("title") is None

    def test_prior_mark_smooth_layer_keeps_the_chart_level_title(self, smooth_df):
        """mark_point().mark_smooth() renders through the layered path,
        whose chart-level encoding (not the rebuilt layer encoding) is the
        title source."""
        chart = (
            fm.Chart(smooth_df)
            .mark_point()
            .mark_smooth(method="lm")
            .encode(x=fm.X("time", title="Time"), y=fm.Y("temp", title="Temp"))
        )
        titles = _axis_title_texts(chart.to_svg())
        assert "Time" in titles
        assert "Temp" in titles


# ---------------------------------------------------------------------------
# mark_function
# ---------------------------------------------------------------------------


class TestMarkFunctionKeepsCallerOptions:
    def test_encode_before_mark_survives(self):
        import numpy as np

        chart = (
            fm.Chart(None)
            .encode(x=fm.X("t", title="Time"), y=fm.Y("v", title="Value"))
            .mark_function(np.sin, domain=[0, 6.28])
        )
        assert chart._encoding["x"].option("title") == "Time"
        assert chart._encoding["y"].option("title") == "Value"
        titles = _axis_title_texts(chart.to_svg())
        assert "Time" in titles
        assert "Value" in titles

    def test_encode_after_mark_overwrites_the_remapped_channel(self):
        """An .encode() call after mark_function() overwrites the remapped
        channel outright."""
        import numpy as np

        chart = (
            fm.Chart(None)
            .mark_function(np.sin, domain=[0, 6.28])
            .encode(x=fm.X("x", title="X"), y=fm.Y("y", title="Y"))
        )
        assert chart._encoding["x"].option("title") == "X"
        assert chart._encoding["y"].option("title") == "Y"
        titles = _axis_title_texts(chart.to_svg())
        assert "X" in titles
        assert "Y" in titles

    def test_layered_with_a_point_chart_keeps_the_point_charts_own_title(self):
        """``point_chart + function_chart`` merges through ``Chart.__add__``,
        which keeps the point chart's own channel options independently of
        ``_apply_remap`` -- which only ever sees the function chart's
        untitled, never-encoded x/y."""
        import numpy as np

        df = pl.DataFrame({"t": [0.0, 1.0, 2.0, 3.0], "v": [0.0, 1.0, 2.0, 3.0]})
        scatter = (
            fm.Chart(df).mark_point().encode(x=fm.X("t", title="Time"), y=fm.Y("v", title="Value"))
        )
        fn_chart = fm.Chart(None).mark_function(np.sin, domain=[0, 6.28])
        titles = _axis_title_texts((scatter + fn_chart).to_svg())
        assert "Time" in titles
        assert "Value" in titles


# ---------------------------------------------------------------------------
# _apply_remap: the discriminating structural checks
# ---------------------------------------------------------------------------


# The field is unchanged under this remap ("bin_start" -> "bin_start"), so
# no title-stamp fires and this set isolates option-preservation on its own
# (the stamp interaction is covered separately below). Restoring only the
# title would drop every one of these except the first.
_CALLER_OPTION_SETS = [
    {"title": "Value"},
    {"axis": fm.Axis(title="AxT")},
    {"format": ".2f"},
    {"format_type": "number"},
    {"scale": fm.LinearScale(domain=[0, 10])},
    {"title": "Value", "format": ".1f", "axis": fm.Axis(grid=False)},
]


@pytest.mark.parametrize("caller_kwargs", _CALLER_OPTION_SETS)
def test_apply_remap_keeps_every_caller_option(caller_kwargs):
    """The remapped channel's options minus {bin, aggregate, type} equal
    the caller's; restoring only the title would fail every case here
    except the first."""
    orig_encoding = {"x": X("bin_start", **caller_kwargs)}
    encoding: dict = {}
    _apply_remap(encoding, {"x": "bin_start"}, orig_encoding=orig_encoding)
    rebuilt_options = {
        k: v for k, v in encoding["x"]._kwargs.items() if k not in ("bin", "aggregate", "type")
    }
    assert rebuilt_options == caller_kwargs


def test_apply_remap_drops_bin_and_aggregate():
    orig_encoding = {
        "x": X("value", bin=True, title="Value"),
        "y": Y("other", aggregate="mean"),
    }
    encoding: dict = {}
    _apply_remap(encoding, {"x": "bin_start", "y": "count"}, orig_encoding=orig_encoding)
    assert "bin" not in encoding["x"]._kwargs
    assert encoding["x"].option("title") == "Value"
    assert "aggregate" not in encoding["y"]._kwargs


def test_apply_remap_forces_quantitative_type_even_when_caller_declared_another():
    orig_encoding = {"x": X("value", type="O", title="Value")}
    encoding: dict = {}
    _apply_remap(encoding, {"x": "bin_start"}, orig_encoding=orig_encoding)
    assert encoding["x"].option("type") == "Q"
    assert encoding["x"].option("title") == "Value"


def test_apply_remap_explicit_title_wins_over_the_field_name_stamp():
    orig_encoding = {"x": X("value", title="Caller Title")}
    encoding: dict = {}
    _apply_remap(encoding, {"x": "bin_start"}, orig_encoding=orig_encoding)
    assert encoding["x"].option("title") == "Caller Title"


def test_apply_remap_stamps_the_original_field_name_when_no_title_given():
    orig_encoding = {"x": X("value")}
    encoding: dict = {}
    _apply_remap(encoding, {"x": "bin_start"}, orig_encoding=orig_encoding)
    assert encoding["x"].option("title") == "value"


def test_apply_remap_no_title_stamp_when_the_field_is_unchanged():
    orig_encoding = {"x": X("x")}
    encoding: dict = {}
    _apply_remap(encoding, {"x": "x"}, orig_encoding=orig_encoding)
    assert encoding["x"].option("title") is None


def test_apply_remap_bare_str_caller_channel_gets_a_fresh_titled_channel():
    """A caller channel given as a bare ``str`` (not wrapped in ``X``/``Y``)
    still recovers its field name as the title when the remap renames it."""
    orig_encoding = {"x": "value"}
    encoding: dict = {}
    _apply_remap(encoding, {"x": "bin_start"}, orig_encoding=orig_encoding)
    assert encoding["x"] == X("bin_start", title="value", type="Q")


def test_apply_remap_bare_str_caller_channel_no_title_stamp_when_unchanged():
    """A bare ``str`` caller channel whose name matches the remap target
    gets no title stamp, the same as a ``ChannelBase`` caller channel."""
    orig_encoding = {"x": "bin_start"}
    encoding: dict = {}
    _apply_remap(encoding, {"x": "bin_start"}, orig_encoding=orig_encoding)
    assert encoding["x"] == X("bin_start", type="Q")


def test_apply_remap_unbound_channel_gets_a_fresh_quantitative_channel():
    encoding: dict = {}
    _apply_remap(encoding, {"y": "count"}, orig_encoding={"x": X("value", title="Value")})
    assert encoding["y"] == Y("count", type="Q")


def test_apply_remap_rebuilds_x2_y2_fresh_regardless_of_caller_options():
    """x2/y2 are desugar-owned spans with no caller options: they are
    always rebuilt fresh from the remap, and the caller's prior x2 (a
    different field entirely) is never consulted."""
    orig_encoding = {"x2": X2("whatever")}
    encoding: dict = {}
    _apply_remap(encoding, {"x2": "bin_end"}, orig_encoding=orig_encoding)
    assert encoding["x2"] == X2("bin_end", type="Q")


def test_apply_remap_with_no_orig_encoding_builds_fresh_channels():
    encoding: dict = {}
    _apply_remap(encoding, {"x": "bin_start", "y": "count"}, orig_encoding=None)
    assert encoding["x"] == X("bin_start", type="Q")
    assert encoding["y"] == Y("count", type="Q")


# ---------------------------------------------------------------------------
# Untitled figure helpers: the field-name default, not a stamped title
# ---------------------------------------------------------------------------
#
# None of these four helpers passes an explicit title on the remap path, so
# each keeps the plain field name as its axis title -- the same default a
# channel with no title ever had, not something a caller's title stamped
# over it.


@pytest.fixture()
def pair_df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0] * 2,
            "b": [8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0] * 2,
        }
    )


@pytest.fixture()
def smooth_fit_df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "x": [float(v) for v in range(1, 11)] * 2,
            "y": [2.0, 2.5, 3.0, 4.1, 4.8, 5.9, 6.7, 7.5, 8.2, 9.9] * 2,
        }
    )


def test_displot_hist_keeps_the_field_name_title(hist_df):
    svg = fm.displot(hist_df, x="measurement").to_svg()
    assert "measurement" in _axis_title_texts(svg)


def test_pairplot_diagonal_keeps_each_fields_own_title(pair_df):
    svg = fm.pairplot(pair_df).to_svg()
    titles = _axis_title_texts(svg)
    assert "a" in titles
    assert "b" in titles


def test_jointplot_marginal_keeps_each_fields_own_title(pair_df):
    svg = fm.jointplot(pair_df, x="a", y="b").to_svg()
    titles = _axis_title_texts(svg)
    assert "a" in titles
    assert "b" in titles


def test_lmplot_fit_keeps_each_fields_own_title(smooth_fit_df):
    svg = fm.lmplot(
        smooth_fit_df, x="x", y="y", ci=None, show_metrics=False, scatter=False
    ).to_svg()
    titles = _axis_title_texts(svg)
    assert "x" in titles
    assert "y" in titles
