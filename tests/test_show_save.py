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


def test_save_html(chart, tmp_path):
    out = tmp_path / "out.html"
    chart.save(out)
    text = out.read_text()
    assert "<!DOCTYPE html>" in text
    assert "ferrum" in text.lower()


def test_save_json(chart, tmp_path):
    import json

    out = tmp_path / "out.json"
    chart.save(out)
    scene = json.loads(out.read_text())
    assert "panels" in scene
    assert "width" in scene


def test_save_unknown_extension_raises(chart, tmp_path):
    with pytest.raises(ValueError, match="extension"):
        chart.save(tmp_path / "out.weird")


def test_save_explicit_format_overrides_extension(chart, tmp_path):
    out = tmp_path / "out.txt"
    chart.save(out, format="svg")
    text = out.read_text()
    assert "<svg" in text or "<?xml" in text


# Task 33: show browser fallback + Jupyter rich-display paths

def test_show_in_non_jupyter_opens_browser(chart, monkeypatch):
    """When not in Jupyter, .show() writes a temp HTML and calls webbrowser.open."""
    opened = []
    monkeypatch.setattr("webbrowser.open", lambda url: opened.append(url))
    monkeypatch.setattr("ferrum.display._is_jupyter", lambda: False)
    chart.show()
    assert len(opened) == 1
    assert opened[0].startswith("file://")
    assert opened[0].endswith(".html")


def test_repr_svg_returns_string_for_jupyter(chart):
    s = chart._repr_svg_()
    assert s is not None
    assert "<svg" in s or "<?xml" in s


def test_repr_html_returns_div_wrapped_svg(chart):
    s = chart._repr_html_()
    assert s is not None
    assert s.startswith("<div>")
