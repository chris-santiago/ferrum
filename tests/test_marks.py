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


def test_deferred_mark_error_for_8b_mark():
    from ferrum.marks import deferred_mark_error, PHASE_8B_MARKS
    e = deferred_mark_error("boxplot")
    assert isinstance(e, NotImplementedError)
    assert "Phase 8b" in str(e)


def test_deferred_mark_error_for_9_plus_mark():
    from ferrum.marks import deferred_mark_error
    e = deferred_mark_error("arc")
    assert "Phase 9+" in str(e)


def test_phase_8b_marks_set_includes_composites_and_heavy_stats():
    from ferrum.marks import PHASE_8B_MARKS
    assert {"boxplot", "errorbar", "violin", "raster"}.issubset(PHASE_8B_MARKS)
