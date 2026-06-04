"""T10: hue-split violins must actually split the KDE per (x, hue) group.

Regression for the flexibility-audit silent-wrong-output bug:
``mark_violin().encode(x=, y=, color="hue:N")`` and ``catplot(kind="violin", hue=...)``
used to collapse to one violin per x-category (ignoring hue) while still drawing a
hue legend. These tests pin the corrected behavior: the ``Violin`` (and inner
``BoxStats``) transform groupby must include BOTH the x field and the hue field, the
violin body layer must carry a ``color`` encoding, and a 2x2 (x, hue) dataset must
render more distinct polygon groups than the no-hue case.
"""

from __future__ import annotations

import json

import polars as pl

import ferrum as fm


def _spec_dict(chart: fm.Chart) -> dict:
    return json.loads(chart.to_spec().to_json())


def _transform(spec: dict, ttype: str) -> dict:
    for t in spec["transforms"]:
        if t["type"] == ttype:
            return t
    raise AssertionError(
        f"transform {ttype!r} not found in {[t['type'] for t in spec['transforms']]}"
    )


def _body_layer(spec: dict) -> dict:
    # The violin body is the first polygon layer reading from the violin source.
    for lyr in spec["layers"]:
        if lyr["mark"] == "polygon":
            return lyr
    raise AssertionError("no polygon (violin body) layer found")


def _make_df() -> pl.DataFrame:
    # 2 x-categories x 2 hue groups with deliberately different distributions per
    # (x, hue) so a split KDE produces visibly different shapes.
    return pl.DataFrame(
        {
            "g": ["a"] * 40 + ["b"] * 40,
            "h": (["x"] * 20 + ["y"] * 20) * 2,
            "v": (
                [float(i) for i in range(20)]  # a/x: spread low
                + [50.0 + float(i) * 0.2 for i in range(20)]  # a/y: tight high
                + [float(i) * 0.3 for i in range(20)]  # b/x: tight low
                + [10.0 + float(i) * 2 for i in range(20)]  # b/y: spread mid
            ),
        }
    )


def test_violin_hue_groupby_includes_both_fields():
    df = _make_df()
    chart = fm.Chart(df).mark_violin(inner="box").encode(x="g:N", y="v:Q", color="h:N")
    spec = _spec_dict(chart)
    violin = _transform(spec, "violin")
    assert violin["groupby"] == ["g", "h"], violin["groupby"]


def test_violin_hue_body_layer_carries_color():
    df = _make_df()
    chart = fm.Chart(df).mark_violin(inner="box").encode(x="g:N", y="v:Q", color="h:N")
    spec = _spec_dict(chart)
    body = _body_layer(spec)
    assert "color" in body["encoding"], body["encoding"]
    assert body["encoding"]["color"]["field"] == "h"


def test_violin_hue_box_stats_groupby_includes_hue():
    df = _make_df()
    chart = fm.Chart(df).mark_violin(inner="box").encode(x="g:N", y="v:Q", color="h:N")
    spec = _spec_dict(chart)
    box = _transform(spec, "box_stats")
    assert box["groupby"] == ["g", "h"], box["groupby"]


def test_violin_hue_quartile_groupby_includes_hue():
    df = _make_df()
    chart = fm.Chart(df).mark_violin(inner="quartile").encode(x="g:N", y="v:Q", color="h:N")
    spec = _spec_dict(chart)
    box = _transform(spec, "box_stats")
    assert box["groupby"] == ["g", "h"], box["groupby"]


def test_violin_hue_renders_more_groups_than_no_hue():
    df = _make_df()
    hue_chart = fm.Chart(df).mark_violin(inner=None).encode(x="g:N", y="v:Q", color="h:N")
    no_hue_chart = fm.Chart(df).mark_violin(inner=None).encode(x="g:N", y="v:Q")

    hue_svg = hue_chart.to_svg()
    no_hue_svg = no_hue_chart.to_svg()

    # Behavioral assertion: the hue split must produce more distinct violin polygon
    # path elements than collapsing to one violin per x-category.
    hue_paths = hue_svg.count("<path")
    no_hue_paths = no_hue_svg.count("<path")
    assert hue_paths > no_hue_paths, (hue_paths, no_hue_paths)


def test_violin_no_hue_back_compat():
    df = _make_df()
    chart = fm.Chart(df).mark_violin(inner="box").encode(x="g:N", y="v:Q")
    spec = _spec_dict(chart)
    violin = _transform(spec, "violin")
    assert violin["groupby"] == ["g"], violin["groupby"]
    box = _transform(spec, "box_stats")
    assert box["groupby"] == ["g"], box["groupby"]
    body = _body_layer(spec)
    assert "color" not in body["encoding"], body["encoding"]


def test_violin_no_hue_byte_identical():
    # The no-hue render must be byte-identical to today (no color threading leakage).
    df = _make_df()
    a = fm.Chart(df).mark_violin(inner="box").encode(x="g:N", y="v:Q").to_svg()
    b = fm.Chart(df).mark_violin(inner="box").encode(x="g:N", y="v:Q").to_svg()
    assert a == b


def test_catplot_violin_hue_routes_through():
    df = _make_df()
    chart = fm.catplot(df, x="g", y="v", hue="h", kind="violin")
    spec = _spec_dict(chart)
    violin = _transform(spec, "violin")
    assert violin["groupby"] == ["g", "h"], violin["groupby"]
    body = _body_layer(spec)
    assert "color" in body["encoding"]
    assert body["encoding"]["color"]["field"] == "h"
