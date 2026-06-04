"""Tests for the value-returning output converters.

Covers the ``to_svg`` / ``to_png`` / ``to_html`` rename, the byte-identity
contract between ``to_html()`` and ``save(".html")``, composition-view
coverage, and the deprecated ``show_svg`` / ``show_png`` aliases.

This file intentionally still exercises the deprecated ``show_*`` names so the
deprecation behavior stays under test; the later suite-wide sweep skips it.
"""

from __future__ import annotations

import polars as pl
import pytest

import ferrum as fm


def _chart() -> fm.Chart:
    df = pl.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
    return fm.Chart(df).mark_point().encode(x="x", y="y")


def test_to_svg_returns_str() -> None:
    chart = _chart()
    svg = chart.to_svg()
    assert isinstance(svg, str)
    assert svg.startswith("<svg")


def test_to_png_returns_png_bytes() -> None:
    chart = _chart()
    png = chart.to_png()
    assert isinstance(png, bytes)
    assert png[:4] == b"\x89PNG"


def test_to_svg_matches_deprecated_show_svg() -> None:
    chart = _chart()
    new = chart.to_svg()
    with pytest.warns(DeprecationWarning):
        old = chart.show_svg()
    assert new == old


def test_to_png_matches_deprecated_show_png() -> None:
    chart = _chart()
    new = chart.to_png()
    with pytest.warns(DeprecationWarning):
        old = chart.show_png()
    assert new == old


def test_to_html_equals_saved_html(tmp_path) -> None:
    chart = _chart()
    html = chart.to_html()
    assert isinstance(html, str)

    dest = tmp_path / "chart.html"
    chart.save(dest)
    assert html == dest.read_text()


def test_composition_to_svg_to_png_to_html(tmp_path) -> None:
    layer = _chart() | _chart()  # HConcatChart

    svg = layer.to_svg()
    assert isinstance(svg, str)
    assert svg.startswith("<svg")

    png = layer.to_png()
    assert isinstance(png, bytes)
    assert png[:4] == b"\x89PNG"

    html = layer.to_html()
    assert isinstance(html, str)
    dest = tmp_path / "layer.html"
    layer.save(dest)
    assert html == dest.read_text()


def test_layer_chart_to_svg() -> None:
    scatter = fm.Chart(pl.DataFrame({"x": [1, 2], "y": [3, 4]})).mark_point().encode(x="x", y="y")
    line = fm.Chart(pl.DataFrame({"x": [1, 2], "y": [3, 4]})).mark_line().encode(x="x", y="y")
    overlay = fm.LayerChart(scatter, line)
    svg = overlay.to_svg()
    assert isinstance(svg, str)
    assert svg.startswith("<svg")


def test_show_svg_deprecated_returns_str() -> None:
    chart = _chart()
    with pytest.warns(DeprecationWarning):
        svg = chart.show_svg()
    assert isinstance(svg, str)
    assert svg.startswith("<svg")


def test_show_png_deprecated_returns_bytes() -> None:
    chart = _chart()
    with pytest.warns(DeprecationWarning):
        png = chart.show_png()
    assert isinstance(png, bytes)
    assert png[:4] == b"\x89PNG"


def test_composition_show_svg_deprecated() -> None:
    layer = _chart() | _chart()
    with pytest.warns(DeprecationWarning):
        svg = layer.show_svg()
    assert isinstance(svg, str)
    assert svg.startswith("<svg")


def test_composition_show_png_deprecated() -> None:
    layer = _chart() | _chart()
    with pytest.warns(DeprecationWarning):
        png = layer.show_png()
    assert isinstance(png, bytes)
    assert png[:4] == b"\x89PNG"
