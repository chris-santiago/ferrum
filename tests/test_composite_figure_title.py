"""Structural tests for composite figure-level chrome consolidation (#8).

`JointChart`, `ClusterMapChart`, and `RepeatChart` historically extended
`_ChartLike` and overrode `.properties()` to forward *all* kwargs (including
``title``/``subtitle``/``caption``) to an inner panel (center / heatmap /
template).  That placed the figure title on a sub-panel instead of the figure.

After consolidation, all composites inherit figure-chrome handling from
`_CompositeBase`:

  - ``.properties(title=, subtitle=, caption=)`` is intercepted at the figure
    level (stored as ``_figure_title`` / ``_figure_subtitle`` /
    ``_figure_caption``) and never fanned to inner panels.
  - Non-chrome kwargs (``width``, ``height``, ...) still reach the inner
    panel(s) they reached before.
  - A canonical title accessor (``_figure_title_text``) resolves the figure
    title for composites and ``_title`` for single charts, so ``to_html``
    sets the document ``<title>`` correctly.

This module covers the structural half (Task 12).  SVG-band and interactive
on-canvas threading are covered separately.
"""

from __future__ import annotations

import polars as pl
import pytest

import ferrum as fm


@pytest.fixture
def df():
    return pl.DataFrame({"x": [1.0, 2.0, 3.0, 4.0], "y": [4.0, 3.0, 2.0, 1.0]})


@pytest.fixture
def base_chart(df):
    return fm.Chart(df).mark_point().encode(x="x", y="y")


@pytest.fixture
def hist_chart(df):
    return fm.Chart(df).mark_bar().encode(x="x", y="y")


# ---------------------------------------------------------------------------
# JointChart
# ---------------------------------------------------------------------------


def test_joint_title_stored_on_figure_not_center(base_chart, hist_chart):
    joint = fm.JointChart(base_chart, top=hist_chart, right=hist_chart)
    out = joint.properties(title="Joint Title")

    assert out._figure_title == "Joint Title"
    assert out._figure_title_text() == "Joint Title"
    # Inner center panel must not have received the title.
    assert out.center._title is None
    assert out.top._title is None
    assert out.right._title is None


def test_joint_subtitle_caption_stored_on_figure(base_chart):
    joint = fm.JointChart(base_chart)
    out = joint.properties(title="T", subtitle="S", caption="C")

    assert out._figure_title == "T"
    assert out._figure_subtitle == "S"
    assert out._figure_caption == "C"
    assert out.center._title is None


def test_joint_non_chrome_kwargs_reach_center(base_chart):
    joint = fm.JointChart(base_chart)
    out = joint.properties(width=320, height=240)

    assert out.center._width == 320
    assert out.center._height == 240
    # No figure chrome was set.
    assert out._figure_title is None


def test_joint_title_plus_width(base_chart):
    joint = fm.JointChart(base_chart)
    out = joint.properties(title="T", width=300)

    assert out._figure_title == "T"
    assert out.center._title is None
    assert out.center._width == 300


# ---------------------------------------------------------------------------
# ClusterMapChart
# ---------------------------------------------------------------------------


def test_clustermap_title_stored_on_figure_not_heatmap(base_chart, hist_chart):
    cm = fm.ClusterMapChart(base_chart, row_dendrogram=hist_chart, col_dendrogram=hist_chart)
    out = cm.properties(title="Cluster Title")

    assert out._figure_title == "Cluster Title"
    assert out._figure_title_text() == "Cluster Title"
    assert out.heatmap._title is None
    assert out.row_dendrogram._title is None
    assert out.col_dendrogram._title is None


def test_clustermap_non_chrome_kwargs_reach_heatmap(base_chart):
    cm = fm.ClusterMapChart(base_chart)
    out = cm.properties(width=500, height=500)

    assert out.heatmap._width == 500
    assert out.heatmap._height == 500
    assert out._figure_title is None


# ---------------------------------------------------------------------------
# RepeatChart
# ---------------------------------------------------------------------------


def test_repeat_title_stored_on_figure_not_template(df):
    template = fm.Chart(df).mark_point().encode(x=fm.Repeat.column, y=fm.Repeat.row)
    rep = fm.RepeatChart(template, row=["x", "y"], column=["x", "y"])
    out = rep.properties(title="Repeat Title")

    assert out._figure_title == "Repeat Title"
    assert out._figure_title_text() == "Repeat Title"
    assert out.template._title is None


def test_repeat_subtitle_caption_stored_on_figure(df):
    template = fm.Chart(df).mark_point().encode(x=fm.Repeat.column, y=fm.Repeat.row)
    rep = fm.RepeatChart(template, row=["x", "y"], column=["x", "y"])
    out = rep.properties(title="T", subtitle="S", caption="C")

    assert out._figure_title == "T"
    assert out._figure_subtitle == "S"
    assert out._figure_caption == "C"
    assert out.template._title is None


def test_repeat_non_chrome_kwargs_reach_template(df):
    template = fm.Chart(df).mark_point().encode(x=fm.Repeat.column, y=fm.Repeat.row)
    rep = fm.RepeatChart(template, row=["x", "y"], column=["x", "y"])
    out = rep.properties(width=200, height=200)

    assert out.template._width == 200
    assert out.template._height == 200
    assert out._figure_title is None


# ---------------------------------------------------------------------------
# Regression: concat composites unchanged
# ---------------------------------------------------------------------------


def test_hconcat_title_stored_on_figure(base_chart, hist_chart):
    out = (base_chart | hist_chart).properties(title="HC")
    assert out._figure_title == "HC"
    assert out._figure_title_text() == "HC"
    for child in out.charts:
        assert child._title is None


def test_vconcat_title_stored_on_figure(base_chart, hist_chart):
    out = (base_chart & hist_chart).properties(title="VC")
    assert out._figure_title == "VC"
    for child in out.charts:
        assert child._title is None


def test_concat_title_stored_on_figure(base_chart, hist_chart):
    out = fm.ConcatChart(base_chart, hist_chart, columns=2).properties(title="CC")
    assert out._figure_title == "CC"
    for child in out.charts:
        assert child._title is None


def test_concat_non_chrome_kwargs_reach_children(base_chart, hist_chart):
    out = fm.ConcatChart(base_chart, hist_chart, columns=2).properties(width=250)
    for child in out.charts:
        assert child._width == 250


# ---------------------------------------------------------------------------
# Canonical accessor + to_html document <title>
# ---------------------------------------------------------------------------


def test_single_chart_accessor_unchanged_via_layer(base_chart):
    # LayerChart keeps reading _title; canonical accessor resolves it.
    layered = fm.LayerChart(base_chart, base_chart, title="Layer T")
    assert layered._figure_title_text() == "Layer T"


def test_layer_accessor_default_when_no_title(base_chart):
    layered = fm.LayerChart(base_chart, base_chart)
    assert layered._figure_title_text() == "Ferrum chart"


def test_joint_to_html_document_title(base_chart):
    joint = fm.JointChart(base_chart).properties(title="Doc Title")
    html = joint.to_html()
    assert "<title>Doc Title</title>" in html


def test_hconcat_to_html_document_title(base_chart, hist_chart):
    out = (base_chart | hist_chart).properties(title="HC Doc")
    html = out.to_html()
    assert "<title>HC Doc</title>" in html


def test_clustermap_to_html_document_title(base_chart):
    cm = fm.ClusterMapChart(base_chart).properties(title="CM Doc")
    html = cm.to_html()
    assert "<title>CM Doc</title>" in html


def test_to_html_default_title_without_figure_title(base_chart):
    joint = fm.JointChart(base_chart)
    html = joint.to_html()
    assert "<title>Ferrum chart</title>" in html
