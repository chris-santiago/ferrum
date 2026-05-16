import polars as pl

from ferrum import Chart
from ferrum._direct_label import _direct_label_endpoint


def test_direct_label_endpoint_emits_text_at_max_x_per_series():
    df = pl.DataFrame(
        {
            "x": [1, 2, 3, 1, 2, 3],
            "y": [0.5, 0.6, 0.7, 0.4, 0.5, 0.5],
            "split": ["train", "train", "train", "val", "val", "val"],
        }
    )
    base = Chart(df).encode(x="x", y="y", color="split").mark_line()
    chart = _direct_label_endpoint(base, label_field="split")
    svg = chart.show_svg()
    assert ">train<" in svg
    assert ">val<" in svg


def test_direct_label_endpoint_bails_when_label_field_missing():
    """Helper returns the original chart untouched when label_field absent."""
    df = pl.DataFrame({"x": [1, 2], "y": [3, 4]})
    base = Chart(df).encode(x="x", y="y").mark_line()
    chart = _direct_label_endpoint(base, label_field="missing")
    # Returned chart is the original — the augmented column is never added.
    assert "_direct_label_text" not in chart._data.columns
