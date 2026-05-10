import polars as pl
import pytest

from ferrum import Chart
from ferrum.annotations import annotate_hline, annotate_vline, annotate_rect, annotate_text


def test_annotate_hline_returns_chart_with_rule_mark():
    h = annotate_hline(0)
    assert h._mark == "rule"


def test_annotate_vline_returns_chart_with_rule_mark():
    v = annotate_vline(5)
    assert v._mark == "rule"


def test_annotate_rect_returns_chart_with_rect_mark():
    r = annotate_rect(0, 1, 0, 1, opacity=0.1)
    assert r._mark == "rect"


def test_annotate_text_returns_chart_with_text_mark():
    t = annotate_text(1.0, 2.0, "hi")
    assert t._mark == "text"


def test_annotate_hline_can_be_added_to_scatter():
    df = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    scatter = Chart(df).mark_point().encode(x="a", y="b")
    # annotate_hline uses different data → falls through to hconcat with warning,
    # OR uses an inline 1-row table that matches the chart's column shape.
    # Phase 8a impl: annotate_* return charts with empty data; the + path
    # detects "same data" check fails → hconcat fallback. That's acceptable for 8a;
    # Phase 9 will improve via a shared-data resolver.
    # For this test, just assert no exception raised.
    with pytest.warns(UserWarning):
        composed = scatter + annotate_hline(5)
    assert composed is not None
