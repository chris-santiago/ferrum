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
    svg = chart.to_svg()
    assert ">train<" in svg
    assert ">val<" in svg


def test_direct_label_endpoint_bails_when_label_field_missing():
    """Helper returns the original chart untouched when label_field absent."""
    df = pl.DataFrame({"x": [1, 2], "y": [3, 4]})
    base = Chart(df).encode(x="x", y="y").mark_line()
    chart = _direct_label_endpoint(base, label_field="missing")
    # Returned chart is the original — the augmented column is never added.
    assert "_direct_label_text" not in chart._data.columns


def test_direct_label_endpoint_skips_null_group_values(recwarn):
    """#220 sibling defect: a null ``label_field`` value (e.g. an injected
    sentinel row) must not reach ``pl.col(label_field) == series``, which
    always evaluates to null in polars and warns "Comparisons with None
    always result in null". The null group's single row is also the one
    series too short to draw, so the render names it (``split=null (one
    point)``)."""
    df = pl.DataFrame(
        {
            "x": [1, 2, 3, 1, 2, 3, 1],
            "y": [0.5, 0.6, 0.7, 0.4, 0.5, 0.5, 0.9],
            "split": ["train", "train", "train", "val", "val", "val", None],
        }
    )
    base = Chart(df).encode(x="x", y="y", color="split").mark_line()
    chart = _direct_label_endpoint(base, label_field="split")
    svg = chart.to_svg()
    assert ">train<" in svg
    assert ">val<" in svg
    user_warnings = [w for w in recwarn.list if issubclass(w.category, UserWarning)]
    expected = (
        "layer 0: mark_line: mark_line() joins two or more points per series, "
        "so it did not draw split=null (one point)"
    )
    assert [str(w.message) for w in user_warnings] == [expected]
