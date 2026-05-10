import pytest

from ferrum.marks.base import MarkBase


def test_markbase_accepts_valid_kwargs():
    m = MarkBase("point", size=100, stroke="#ff0000", opacity=0.5)
    assert m._kwargs == {"size": 100, "stroke": "#ff0000", "opacity": 0.5}


def test_markbase_rejects_unknown_kwargs():
    with pytest.raises(TypeError, match="unknown keyword"):
        MarkBase("point", squiggly=True)


def test_to_mark_kwargs_dict_filters_to_style_only():
    m = MarkBase("smooth", size=100, method="loess", bandwidth=0.5)
    d = m.to_mark_kwargs_dict()
    assert d == {"size": 100}   # method and bandwidth go to transforms, not style
