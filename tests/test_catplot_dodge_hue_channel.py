"""Net-new coverage: ``catplot(hue=<channel object>, dodge=True)`` must render.

``_catplot_build`` built its dodge adjustment as ``Dodge(by=hue)`` from the
RAW ``hue=`` argument. ``Dodge(by=)`` takes a column name, so a documented
channel-object spelling (``hue=fm.Color("g")``) put the channel object itself
on the wire and the call died in serialization with an untyped
``TypeError: Object of type Color is not JSON serializable`` -- naming neither
``catplot`` nor ``dodge=``, and offering the caller nothing to act on.

This was the fourth site of a class already closed at three others in the same
file (``_catplot_summary_layers``'s ``hue_field``, the ``kind="count"``
groupby, ``_displot_build``'s ``hue_field``), each by resolving the argument
through ``_field_name`` exactly once. The dodge adjustment now does the same.

Scoped to that one defect per this repo's Python test-file convention (see
``test_boxen_palette.py``'s module docstring).
"""

from __future__ import annotations

import polars as pl
import pytest

import ferrum as fm

#: Every catplot kind whose mark call passes the dodge adjustment through.
_DODGING_KINDS = ("strip", "swarm", "box", "violin", "boxen", "point", "bar")


@pytest.fixture
def cat_df():
    return pl.DataFrame(
        {
            "cat": ["a", "b", "c", "d"] * 10,
            "val": [1.0, 2.0, 3.0, 4.0, 2.5, 1.5, 3.5, 4.5] * 5,
            "grp": ["X", "Y"] * 20,
        }
    )


@pytest.mark.parametrize("kind", _DODGING_KINDS)
def test_channel_object_hue_dodges_like_the_bare_string(cat_df, kind):
    """A channel-object hue renders, and renders identically to the bare
    string it wraps -- the channel is unwrapped for `by=`, not dropped."""
    bare = fm.catplot(cat_df, x="cat", y="val", hue="grp", kind=kind, dodge=True).to_svg()
    channel = fm.catplot(
        cat_df, x="cat", y="val", hue=fm.Color("grp"), kind=kind, dodge=True
    ).to_svg()
    assert "NaN" not in channel
    assert channel == bare


def test_count_kind_channel_object_hue_dodges(cat_df):
    """`kind="count"` takes no `y=`; it reaches the same adjustment."""
    bare = fm.catplot(cat_df, x="cat", hue="grp", kind="count", dodge=True).to_svg()
    channel = fm.catplot(cat_df, x="cat", hue=fm.Color("grp"), kind="count", dodge=True).to_svg()
    assert channel == bare


def test_dodge_position_names_the_hue_column(cat_df):
    """The wire states the column, not the channel -- what the raw object
    reaching serialization could never do."""
    chart = fm.catplot(cat_df, x="cat", y="val", hue=fm.Color("grp"), kind="box", dodge=True)
    assert chart.to_dict()["position"] == {"type": "dodge", "by": "grp", "padding": 0.05}


def test_channel_options_still_reach_the_legend(cat_df):
    """Unwrapping for `by=` must not cost the channel its own options: the
    caller's `sort=` still orders the rendered legend."""
    svg = fm.catplot(
        cat_df,
        x="cat",
        y="val",
        hue=fm.Color("grp", sort=["Y", "X"]),
        kind="box",
        dodge=True,
    ).to_svg()
    labels = [t for t in ("X", "Y") if f">{t}<" in svg]
    assert labels, "no legend labels rendered"
    assert svg.index(">Y<") < svg.index(">X<"), "the channel's own sort= was lost"
