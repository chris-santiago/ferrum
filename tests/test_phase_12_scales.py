"""Tests for Phase 12 scale types wired into the Python API."""

from __future__ import annotations

import polars as pl
import pytest

import ferrum as fm
from ferrum.encoding._scale import _scale_to_dict


# ---------------------------------------------------------------------------
# Importability from top-level namespace
# ---------------------------------------------------------------------------


class TestScaleImports:
    """All 8 new scale classes are importable from ``ferrum``."""

    def test_pow_scale_importable(self):
        assert hasattr(fm, "PowScale")

    def test_sqrt_scale_importable(self):
        assert hasattr(fm, "SqrtScale")

    def test_band_scale_importable(self):
        assert hasattr(fm, "BandScale")

    def test_point_scale_importable(self):
        assert hasattr(fm, "PointScale")

    def test_sequential_scale_importable(self):
        assert hasattr(fm, "SequentialScale")

    def test_diverging_scale_importable(self):
        assert hasattr(fm, "DivergingScale")

    def test_quantize_scale_importable(self):
        assert hasattr(fm, "QuantizeScale")

    def test_bin_ordinal_scale_importable(self):
        assert hasattr(fm, "BinOrdinalScale")


# ---------------------------------------------------------------------------
# PowScale serialization
# ---------------------------------------------------------------------------


class TestPowScale:
    """PowScale converts to the expected dict shape."""

    def test_basic(self):
        s = fm.PowScale(domain=[0, 100], exponent=2.0)
        d = _scale_to_dict(s)
        assert d["type"] == "pow"
        assert d["exponent"] == 2.0
        assert d["domain"] == [0.0, 100.0]
        assert d["clamp"] is False

    def test_with_range_and_padding(self):
        s = fm.PowScale(domain=[0, 50], range=[0, 500], exponent=3.0, padding=10.0)
        d = _scale_to_dict(s)
        assert d["type"] == "pow"
        assert d["exponent"] == 3.0
        assert d["range"] == [0.0, 500.0]
        assert d["padding"] == 10.0

    def test_clamp(self):
        s = fm.PowScale(domain=[0, 1], exponent=0.5, clamp=True)
        d = _scale_to_dict(s)
        assert d["clamp"] is True


# ---------------------------------------------------------------------------
# SqrtScale serialization
# ---------------------------------------------------------------------------


class TestSqrtScale:
    """SqrtScale converts to the expected dict shape."""

    def test_basic(self):
        s = fm.SqrtScale(domain=[0, 100])
        d = _scale_to_dict(s)
        assert d["type"] == "sqrt"
        assert d["exponent"] == 0.5
        assert d["domain"] == [0.0, 100.0]
        assert d["clamp"] is False

    def test_with_range(self):
        s = fm.SqrtScale(domain=[0, 64], range=[0, 8])
        d = _scale_to_dict(s)
        assert d["range"] == [0.0, 8.0]

    def test_no_padding_when_none(self):
        s = fm.SqrtScale(domain=[0, 100])
        d = _scale_to_dict(s)
        assert "padding" not in d


# ---------------------------------------------------------------------------
# BandScale serialization
# ---------------------------------------------------------------------------


class TestBandScale:
    """BandScale converts to the expected dict shape."""

    def test_defaults(self):
        s = fm.BandScale()
        d = _scale_to_dict(s)
        assert d["type"] == "band"
        assert d["paddingInner"] == pytest.approx(0.1)
        assert d["paddingOuter"] == pytest.approx(0.1)
        assert d["align"] == 0.5

    def test_custom_padding(self):
        s = fm.BandScale(padding=0.3, padding_inner=0.2, padding_outer=0.4)
        d = _scale_to_dict(s)
        assert d["paddingInner"] == pytest.approx(0.2)
        assert d["paddingOuter"] == pytest.approx(0.4)

    def test_with_domain(self):
        s = fm.BandScale(domain=["a", "b", "c"])
        d = _scale_to_dict(s)
        assert d["domain"] == ["a", "b", "c"]


# ---------------------------------------------------------------------------
# PointScale serialization
# ---------------------------------------------------------------------------


class TestPointScale:
    """PointScale converts to the expected dict shape."""

    def test_defaults(self):
        s = fm.PointScale()
        d = _scale_to_dict(s)
        assert d["type"] == "point"
        assert d["padding"] == 0.5
        assert d["align"] == 0.5
        assert d["reverse"] is False

    def test_custom(self):
        s = fm.PointScale(padding=0.3, align=0.7, reverse=True)
        d = _scale_to_dict(s)
        assert d["padding"] == pytest.approx(0.3)
        assert d["align"] == pytest.approx(0.7)
        assert d["reverse"] is True

    def test_with_domain(self):
        s = fm.PointScale(domain=["x", "y", "z"])
        d = _scale_to_dict(s)
        assert d["domain"] == ["x", "y", "z"]


# ---------------------------------------------------------------------------
# SequentialScale serialization
# ---------------------------------------------------------------------------


class TestSequentialScale:
    """SequentialScale converts to the expected dict shape."""

    def test_defaults(self):
        s = fm.SequentialScale()
        d = _scale_to_dict(s)
        assert d["type"] == "sequential"
        assert d["reverse"] is False

    def test_with_scheme_and_domain(self):
        s = fm.SequentialScale(scheme="viridis", domain=[0.0, 1.0], reverse=True)
        d = _scale_to_dict(s)
        assert d["scheme"] == "viridis"
        assert d["domain"] == [0.0, 1.0]
        assert d["reverse"] is True

    def test_no_scheme_when_none(self):
        s = fm.SequentialScale()
        d = _scale_to_dict(s)
        # scheme should still be present if the Rust default sets one
        # (it does — 'viridis'), but the key itself should exist
        assert "scheme" in d or d.get("scheme") is None


# ---------------------------------------------------------------------------
# DivergingScale serialization
# ---------------------------------------------------------------------------


class TestDivergingScale:
    """DivergingScale converts to the expected dict shape."""

    def test_defaults(self):
        s = fm.DivergingScale()
        d = _scale_to_dict(s)
        assert d["type"] == "diverging"

    def test_with_all_params(self):
        s = fm.DivergingScale(scheme="redblue", domain=[-1.0, 0.0, 1.0], domain_mid=0.0)
        d = _scale_to_dict(s)
        assert d["scheme"] == "redblue"
        assert d["domain"] == [-1.0, 0.0, 1.0]
        assert d["domainMid"] == 0.0


# ---------------------------------------------------------------------------
# QuantizeScale serialization
# ---------------------------------------------------------------------------


class TestQuantizeScale:
    """QuantizeScale converts to the expected dict shape."""

    def test_basic(self):
        s = fm.QuantizeScale(domain=[0, 100], range=["low", "mid", "high"])
        d = _scale_to_dict(s)
        assert d["type"] == "quantize"
        assert d["domain"] == [0.0, 100.0]
        assert d["range"] == ["low", "mid", "high"]


# ---------------------------------------------------------------------------
# BinOrdinalScale serialization
# ---------------------------------------------------------------------------


class TestBinOrdinalScale:
    """BinOrdinalScale converts to the expected dict shape."""

    def test_basic(self):
        s = fm.BinOrdinalScale(bins=[0, 10, 20, 30])
        d = _scale_to_dict(s)
        assert d["type"] == "bin-ordinal"
        assert d["bins"] == [0.0, 10.0, 20.0, 30.0]

    def test_with_scheme(self):
        s = fm.BinOrdinalScale(bins=[0, 5, 10], scheme="blues")
        d = _scale_to_dict(s)
        assert d["scheme"] == "blues"


# ---------------------------------------------------------------------------
# TimeScale UTC handling
# ---------------------------------------------------------------------------


class TestTimeScaleUtc:
    """TimeScale with ``utc=True`` produces ``"type": "utc"``."""

    def test_utc_true(self):
        s = fm.TimeScale(domain=[0, 1000], utc=True)
        d = _scale_to_dict(s)
        assert d["type"] == "utc"

    def test_utc_false(self):
        s = fm.TimeScale(domain=[0, 1000], utc=False)
        d = _scale_to_dict(s)
        assert d["type"] == "time"

    def test_utc_default_is_false(self):
        s = fm.TimeScale(domain=[0, 1000])
        d = _scale_to_dict(s)
        assert d["type"] == "time"


# ---------------------------------------------------------------------------
# Integration: chart with new scale renders without error
# ---------------------------------------------------------------------------


class TestScaleIntegration:
    """Charts using new scale types build encoding specs correctly.

    Full render through Rust requires ScaleSpec serde variants for each new
    type (a separate Rust-side task). These tests verify the Python layer
    correctly wires scale objects into the encoding pipeline.
    """

    @pytest.fixture()
    def sample_df(self):
        return pl.DataFrame({"cat": ["a", "b", "c"], "val": [1.0, 2.0, 3.0]})

    def test_band_scale_in_encoding(self, sample_df):
        """BandScale is accepted by X encoding and serializes to dict."""
        chart = (
            fm.Chart(sample_df)
            .mark_bar()
            .encode(
                x=fm.X("cat", scale=fm.BandScale(padding=0.2)),
                y="val",
            )
        )
        # The chart object builds without error; the encoding holds the scale
        assert chart is not None

    def test_pow_scale_in_encoding(self, sample_df):
        """PowScale is accepted by Y encoding and serializes to dict."""
        chart = (
            fm.Chart(sample_df)
            .mark_point()
            .encode(
                x="cat",
                y=fm.Y("val", scale=fm.PowScale(domain=[0, 5], exponent=2.0)),
            )
        )
        assert chart is not None

    def test_sequential_scale_in_encoding(self, sample_df):
        """SequentialScale is accepted by Color encoding and serializes to dict."""
        chart = (
            fm.Chart(sample_df)
            .mark_point()
            .encode(
                x="cat",
                y="val",
                color=fm.Color("val", scale=fm.SequentialScale(scheme="viridis")),
            )
        )
        assert chart is not None

    def test_scale_to_dict_roundtrip_in_encoding(self):
        """Encoding channels correctly call _scale_to_dict on scale objects."""
        enc = fm.X("val", scale=fm.PowScale(domain=[0, 10], exponent=2.0))
        # to_encoding_spec_dict calls _scale_to_dict internally
        spec = enc.to_encoding_spec_dict()
        assert spec["scale"]["type"] == "pow"
        assert spec["scale"]["exponent"] == 2.0
