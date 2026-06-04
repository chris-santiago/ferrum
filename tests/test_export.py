"""Tests for F13 (PNG scale parameter) and F14 (PDF export)."""

from __future__ import annotations

import struct

import polars as pl
import pytest

import ferrum as fm


@pytest.fixture()
def simple_chart():
    df = pl.DataFrame({"x": [1.0, 2.0, 3.0], "y": [4.0, 5.0, 6.0]})
    return fm.Chart(df).mark_point().encode(x="x", y="y")


# ---------------------------------------------------------------------------
# F13: PNG scale parameter — Chart._RenderMixin
# ---------------------------------------------------------------------------


def test_to_png_default_scale(simple_chart):
    """to_png() with default scale=2.0 returns valid PNG bytes."""
    png = simple_chart.to_png()
    assert png[:8] == b"\x89PNG\r\n\x1a\n", "expected PNG magic bytes"


def test_to_png_scale_param_explicit_default(simple_chart):
    """to_png(scale=2.0) equals the no-arg default."""
    default = simple_chart.to_png()
    explicit = simple_chart.to_png(scale=2.0)
    # Both should be valid PNG; they should be identical (deterministic render)
    assert default[:8] == b"\x89PNG\r\n\x1a\n"
    assert explicit[:8] == b"\x89PNG\r\n\x1a\n"
    assert default == explicit


def test_to_png_scale_1x_smaller_than_2x(simple_chart):
    """1x PNG is smaller than 2x PNG (fewer pixels)."""
    png_1x = simple_chart.to_png(scale=1.0)
    png_2x = simple_chart.to_png(scale=2.0)
    assert png_1x[:8] == b"\x89PNG\r\n\x1a\n"
    assert png_2x[:8] == b"\x89PNG\r\n\x1a\n"
    assert len(png_1x) < len(png_2x)


def test_to_png_scale_3x_larger_than_2x(simple_chart):
    """3x PNG is larger than 2x PNG."""
    png_2x = simple_chart.to_png(scale=2.0)
    png_3x = simple_chart.to_png(scale=3.0)
    assert len(png_3x) > len(png_2x)


def test_to_png_scale_pixel_dimensions(simple_chart):
    """PNG dimensions reflect the scale factor."""
    png_1x = simple_chart.to_png(scale=1.0)
    png_2x = simple_chart.to_png(scale=2.0)

    # PNG IHDR chunk starts at byte 8: 4-byte length, 4-byte "IHDR",
    # then 4-byte width, 4-byte height
    def _png_dims(png: bytes) -> tuple[int, int]:
        w = struct.unpack(">I", png[16:20])[0]
        h = struct.unpack(">I", png[20:24])[0]
        return w, h

    w1, h1 = _png_dims(png_1x)
    w2, h2 = _png_dims(png_2x)
    assert w2 == w1 * 2
    assert h2 == h1 * 2


# ---------------------------------------------------------------------------
# F13: PNG scale parameter — _ChartLike (composition)
# ---------------------------------------------------------------------------


def test_composition_to_png_scale_param():
    """_ChartLike.to_png(scale=...) propagates scale."""
    df = pl.DataFrame({"x": [1.0, 2.0], "y": [3.0, 4.0]})
    c = fm.Chart(df).mark_point().encode(x="x", y="y")
    comp = c | c  # HConcat composition

    png_1x = comp.to_png(scale=1.0)
    png_2x = comp.to_png(scale=2.0)
    assert png_1x[:8] == b"\x89PNG\r\n\x1a\n"
    assert len(png_1x) < len(png_2x)


def test_composition_to_png_default_unchanged():
    """_ChartLike.to_png() default (scale=2.0) returns valid PNG."""
    df = pl.DataFrame({"x": [1.0, 2.0], "y": [3.0, 4.0]})
    c = fm.Chart(df).mark_point().encode(x="x", y="y")
    comp = c | c
    png = comp.to_png()
    assert png[:8] == b"\x89PNG\r\n\x1a\n"


# ---------------------------------------------------------------------------
# F13: save() scale kwarg for PNG
# ---------------------------------------------------------------------------


def test_save_png_scale_kwarg(simple_chart, tmp_path):
    """save() passes scale to PNG rendering."""
    p1 = str(tmp_path / "chart_1x.png")
    p2 = str(tmp_path / "chart_2x.png")
    simple_chart.save(p1, scale=1.0)
    simple_chart.save(p2, scale=2.0)
    data_1x = (tmp_path / "chart_1x.png").read_bytes()
    data_2x = (tmp_path / "chart_2x.png").read_bytes()
    assert data_1x[:8] == b"\x89PNG\r\n\x1a\n"
    assert len(data_1x) < len(data_2x)


def test_save_png_scale_not_passed_to_html(simple_chart, tmp_path):
    """scale kwarg is silently ignored for non-PNG formats (no error)."""
    p = str(tmp_path / "chart.svg")
    # should not raise even though scale is not applicable
    simple_chart.save(p, scale=2.0)
    assert (tmp_path / "chart.svg").exists()


# ---------------------------------------------------------------------------
# F14: PDF export — save_chart / Chart.save
# ---------------------------------------------------------------------------


def test_save_pdf_produces_pdf_header(simple_chart, tmp_path):
    """Saving as PDF writes a file starting with %PDF-."""
    path = str(tmp_path / "chart.pdf")
    simple_chart.save(path)
    data = (tmp_path / "chart.pdf").read_bytes()
    assert data[:5] == b"%PDF-", f"expected PDF header, got {data[:10]!r}"


def test_save_pdf_format_explicit(simple_chart, tmp_path):
    """format='pdf' override also produces a valid PDF."""
    path = str(tmp_path / "chart.noext")
    simple_chart.save(path, format="pdf")
    data = (tmp_path / "chart.noext").read_bytes()
    assert data[:5] == b"%PDF-"


def test_save_pdf_is_not_empty(simple_chart, tmp_path):
    """PDF output has substantial content (not just a header stub)."""
    path = str(tmp_path / "chart.pdf")
    simple_chart.save(path)
    size = (tmp_path / "chart.pdf").stat().st_size
    assert size > 1024, f"PDF too small ({size} bytes) — may be corrupt"


def test_save_pdf_composition(tmp_path):
    """Composition types can also save as PDF."""
    df = pl.DataFrame({"x": [1.0, 2.0], "y": [3.0, 4.0]})
    c = fm.Chart(df).mark_point().encode(x="x", y="y")
    comp = c | c
    path = str(tmp_path / "comp.pdf")
    comp.save(path)
    data = (tmp_path / "comp.pdf").read_bytes()
    assert data[:5] == b"%PDF-"


def test_save_unknown_extension_still_raises(simple_chart, tmp_path):
    """Unknown extensions still raise ValueError."""
    path = str(tmp_path / "chart.xyz")
    with pytest.raises(ValueError, match="xyz"):
        simple_chart.save(path)
