"""Bar chart y-axis zero-anchoring default."""

import json

import polars as pl
import pytest

import ferrum as fm


def _enc(chart, channel):
    """Return the encoding dict for `channel` from chart.to_spec().to_json()."""
    d = json.loads(chart.to_spec().to_json())
    return d.get("encoding", {}).get(channel, {})


def test_bar_default_zero_anchor():
    """mark_bar injects scale.zero=True on y-encoding by default."""
    df = pl.DataFrame({"cat": ["a", "b", "c"], "val": [10, 20, 15]})
    chart = fm.Chart(df).mark_bar().encode(x="cat", y="val")
    scale = _enc(chart, "y").get("scale", {})
    assert scale.get("zero") is True


def test_bar_explicit_domain_no_zero():
    """User-supplied domain suppresses the zero injection."""
    df = pl.DataFrame({"cat": ["a", "b", "c"], "val": [10, 20, 15]})
    chart = fm.Chart(df).mark_bar().encode(x="cat", y=fm.Y("val", scale={"domain": [5, 25]}))
    scale = _enc(chart, "y").get("scale", {})
    assert "zero" not in scale or scale.get("zero") is not True
    assert scale.get("domain") == [5, 25]


def test_bar_explicit_zero_false():
    """User can opt out of zero-anchoring."""
    df = pl.DataFrame({"cat": ["a", "b", "c"], "val": [10, 20, 15]})
    chart = fm.Chart(df).mark_bar().encode(x="cat", y=fm.Y("val", scale={"zero": False}))
    scale = _enc(chart, "y").get("scale", {})
    assert scale.get("zero") is False


def test_non_bar_mark_no_zero_injection():
    """Non-bar marks do not get the zero injection on y."""
    df = pl.DataFrame({"x": [1, 2, 3], "y": [10, 20, 15]})
    chart = fm.Chart(df).mark_line().encode(x="x", y="y")
    scale = _enc(chart, "y").get("scale", {})
    assert "zero" not in scale
