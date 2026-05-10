import tempfile
from pathlib import Path

import polars as pl
import pytest

from ferrum import Chart


@pytest.fixture
def chart():
    df = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    return Chart(df).mark_point().encode(x="a", y="b")


def test_save_svg(chart, tmp_path):
    out = tmp_path / "out.svg"
    chart.save(out)
    text = out.read_text()
    assert text.startswith("<svg") or text.startswith("<?xml")


def test_save_png(chart, tmp_path):
    out = tmp_path / "out.png"
    chart.save(out)
    bytes_ = out.read_bytes()
    assert bytes_.startswith(b"\x89PNG\r\n\x1a\n")


def test_save_html_raises_not_implemented(chart, tmp_path):
    with pytest.raises(NotImplementedError, match="html"):
        chart.save(tmp_path / "out.html")


def test_save_json_raises_not_implemented(chart, tmp_path):
    with pytest.raises(NotImplementedError, match="json"):
        chart.save(tmp_path / "out.json")


def test_save_unknown_extension_raises(chart, tmp_path):
    with pytest.raises(ValueError, match="extension"):
        chart.save(tmp_path / "out.weird")


def test_save_explicit_format_overrides_extension(chart, tmp_path):
    out = tmp_path / "out.txt"
    chart.save(out, format="svg")
    text = out.read_text()
    assert "<svg" in text or "<?xml" in text
