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


def test_desugar_density_returns_area_with_kde_transform():
    from ferrum.marks.statistical import desugar_density
    from ferrum import Kde
    mark, transforms, remap = desugar_density("price")
    assert mark == "area"
    assert len(transforms) == 1 and isinstance(transforms[0], Kde)
    assert remap == {"y": "density"}


def test_desugar_histogram_returns_bar_with_bin_transform():
    from ferrum.marks.statistical import desugar_histogram
    from ferrum import Bin
    mark, transforms, remap = desugar_histogram("price", bin_count=20)
    assert mark == "bar"
    assert isinstance(transforms[0], Bin)
    assert remap == {"x": "bin_start", "x2": "bin_end", "y": "count"}


def test_desugar_smooth_warns_on_ci_kwarg():
    import warnings
    from ferrum._warn import reset_warnings
    from ferrum.marks.statistical import desugar_smooth

    reset_warnings()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        desugar_smooth("x_col", "y_col", ci=0.95)
    assert any("ci=" in str(wi.message) and "Phase 8b" in str(wi.message) for wi in w)
