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


def test_annotate_rect_encodes_x2_y2():
    """BUG-2 regression: annotate_rect must encode x2 and y2 channels."""
    r = annotate_rect(1.0, 3.0, 2.0, 4.0)
    enc = r._encoding
    assert "x2" in enc, "annotate_rect must encode x2"
    assert "y2" in enc, "annotate_rect must encode y2"


def test_annotate_text_encodes_text_channel():
    """BUG-3 regression: annotate_text must encode the text channel."""
    t = annotate_text(1.0, 2.0, "hello")
    enc = t._encoding
    assert "text" in enc, "annotate_text must encode the text channel"


def test_annotate_hline_can_be_added_to_scatter():
    df = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    scatter = Chart(df).mark_point().encode(x="a", y="b")
    # + always layers now — annotate_hline's different data is auto null-pad
    # merged into the scatter's DataFrame.
    composed = scatter + annotate_hline(5)
    assert composed is not None
    assert composed._layers is not None
