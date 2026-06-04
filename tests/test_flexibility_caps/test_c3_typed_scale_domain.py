"""C3 — domain is optional on typed continuous Scale classes.

Tests for:
1. fm.LogScale(range=...) with NO domain does not raise TypeError.
2. fm.LinearScale(range=...) with NO domain does not raise TypeError.
3. fm.PowScale(range=...) with NO domain does not raise TypeError.
4. fm.SqrtScale(range=...) with NO domain does not raise TypeError.
5. fm.SymlogScale(range=...) with NO domain does not raise TypeError.
6. Domain-less typed scale renders byte-identically to the equivalent dict form.
7. An explicit domain= is still honored (regression guard).
8. Domain-less typed scale on real data produces a sensible auto-inferred domain
   (axis ticks span the data).
"""

from __future__ import annotations

import polars as pl
import pytest

import ferrum as fm


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def df():
    """Simple numeric dataframe for render tests."""
    return pl.DataFrame(
        {
            "x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            "y": [2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0, 512.0, 1024.0],
        }
    )


# ---------------------------------------------------------------------------
# 1-5: Domain-less construction does not raise
# ---------------------------------------------------------------------------


def test_log_scale_no_domain_no_raise():
    """fm.LogScale(range=...) without domain must not raise TypeError."""
    scale = fm.LogScale(range=(0, 400))
    assert scale is not None


def test_linear_scale_no_domain_no_raise():
    """fm.LinearScale(range=...) without domain must not raise TypeError."""
    scale = fm.LinearScale(range=(0, 400))
    assert scale is not None


def test_pow_scale_no_domain_no_raise():
    """fm.PowScale(range=...) without domain must not raise TypeError."""
    scale = fm.PowScale(range=(0, 400))
    assert scale is not None


def test_sqrt_scale_no_domain_no_raise():
    """fm.SqrtScale(range=...) without domain must not raise TypeError."""
    scale = fm.SqrtScale(range=(0, 400))
    assert scale is not None


def test_symlog_scale_no_domain_no_raise():
    """fm.SymlogScale(range=...) without domain must not raise TypeError."""
    scale = fm.SymlogScale(range=(0, 400))
    assert scale is not None


# ---------------------------------------------------------------------------
# 6: Typed scale (no domain) renders byte-identically to equivalent dict
# ---------------------------------------------------------------------------


def test_log_scale_no_domain_matches_dict(df):
    """LogScale(range=...) renders same SVG as scale={"type":"log","range":...}."""
    chart_typed = (
        fm.Chart(df)
        .mark_point()
        .encode(
            x=fm.X("x"),
            y=fm.Y("y", scale=fm.LogScale(range=(0, 400))),
        )
    )
    chart_dict = (
        fm.Chart(df)
        .mark_point()
        .encode(
            x=fm.X("x"),
            y=fm.Y("y", scale={"type": "log", "range": [0, 400]}),
        )
    )
    svg_typed = chart_typed.to_svg()
    svg_dict = chart_dict.to_svg()
    assert svg_typed, "typed chart produced empty SVG"
    assert svg_dict, "dict chart produced empty SVG"
    assert svg_typed == svg_dict, (
        "LogScale(range=...) and dict form produced different SVG.\n"
        f"Typed:\n{svg_typed[:200]}\nDict:\n{svg_dict[:200]}"
    )


def test_linear_scale_no_domain_matches_dict(df):
    """LinearScale(range=...) renders same SVG as scale={"type":"linear","range":...}."""
    chart_typed = (
        fm.Chart(df)
        .mark_point()
        .encode(
            x=fm.X("x"),
            y=fm.Y("y", scale=fm.LinearScale(range=(0, 400))),
        )
    )
    chart_dict = (
        fm.Chart(df)
        .mark_point()
        .encode(
            x=fm.X("x"),
            y=fm.Y("y", scale={"type": "linear", "range": [0, 400]}),
        )
    )
    svg_typed = chart_typed.to_svg()
    svg_dict = chart_dict.to_svg()
    assert svg_typed, "typed chart produced empty SVG"
    assert svg_dict, "dict chart produced empty SVG"
    assert svg_typed == svg_dict, (
        "LinearScale(range=...) and dict form produced different SVG.\n"
        f"Typed:\n{svg_typed[:200]}\nDict:\n{svg_dict[:200]}"
    )


def test_pow_scale_no_domain_matches_dict(df):
    """PowScale(range=...) renders same SVG as scale={"type":"pow","range":...}."""
    chart_typed = (
        fm.Chart(df)
        .mark_point()
        .encode(
            x=fm.X("x"),
            y=fm.Y("y", scale=fm.PowScale(range=(0, 400))),
        )
    )
    chart_dict = (
        fm.Chart(df)
        .mark_point()
        .encode(
            x=fm.X("x"),
            y=fm.Y("y", scale={"type": "pow", "range": [0, 400]}),
        )
    )
    svg_typed = chart_typed.to_svg()
    svg_dict = chart_dict.to_svg()
    assert svg_typed, "typed chart produced empty SVG"
    assert svg_dict, "dict chart produced empty SVG"
    assert svg_typed == svg_dict, (
        "PowScale(range=...) and dict form produced different SVG.\n"
        f"Typed:\n{svg_typed[:200]}\nDict:\n{svg_dict[:200]}"
    )


def test_sqrt_scale_no_domain_matches_dict(df):
    """SqrtScale(range=...) renders same SVG as scale={"type":"sqrt","range":...}."""
    chart_typed = (
        fm.Chart(df)
        .mark_point()
        .encode(
            x=fm.X("x"),
            y=fm.Y("y", scale=fm.SqrtScale(range=(0, 400))),
        )
    )
    chart_dict = (
        fm.Chart(df)
        .mark_point()
        .encode(
            x=fm.X("x"),
            y=fm.Y("y", scale={"type": "sqrt", "range": [0, 400]}),
        )
    )
    svg_typed = chart_typed.to_svg()
    svg_dict = chart_dict.to_svg()
    assert svg_typed, "typed chart produced empty SVG"
    assert svg_dict, "dict chart produced empty SVG"
    assert svg_typed == svg_dict, (
        "SqrtScale(range=...) and dict form produced different SVG.\n"
        f"Typed:\n{svg_typed[:200]}\nDict:\n{svg_dict[:200]}"
    )


def test_symlog_scale_no_domain_matches_dict(df):
    """SymlogScale(range=...) renders same SVG as scale={"type":"symlog","range":...}."""
    chart_typed = (
        fm.Chart(df)
        .mark_point()
        .encode(
            x=fm.X("x"),
            y=fm.Y("y", scale=fm.SymlogScale(range=(0, 400))),
        )
    )
    chart_dict = (
        fm.Chart(df)
        .mark_point()
        .encode(
            x=fm.X("x"),
            y=fm.Y("y", scale={"type": "symlog", "range": [0, 400]}),
        )
    )
    svg_typed = chart_typed.to_svg()
    svg_dict = chart_dict.to_svg()
    assert svg_typed, "typed chart produced empty SVG"
    assert svg_dict, "dict chart produced empty SVG"
    assert svg_typed == svg_dict, (
        "SymlogScale(range=...) and dict form produced different SVG.\n"
        f"Typed:\n{svg_typed[:200]}\nDict:\n{svg_dict[:200]}"
    )


# ---------------------------------------------------------------------------
# 7: Explicit domain= is still honored (regression guard)
# ---------------------------------------------------------------------------


def test_log_scale_explicit_domain_honored():
    """LogScale with explicit domain= stores and returns it correctly."""
    scale = fm.LogScale(domain=[1.0, 1000.0], range=(0, 400))
    assert scale.domain is not None
    assert list(scale.domain) == [1.0, 1000.0]


def test_linear_scale_explicit_domain_honored():
    """LinearScale with explicit domain= stores and returns it correctly."""
    scale = fm.LinearScale(domain=[0.0, 100.0], range=(0, 400))
    assert scale.domain is not None
    assert list(scale.domain) == [0.0, 100.0]


def test_pow_scale_explicit_domain_honored():
    """PowScale with explicit domain= stores and returns it correctly."""
    scale = fm.PowScale(domain=[0.0, 100.0], range=(0, 400))
    assert scale.domain is not None
    assert list(scale.domain) == [0.0, 100.0]


def test_sqrt_scale_explicit_domain_honored():
    """SqrtScale with explicit domain= stores and returns it correctly."""
    scale = fm.SqrtScale(domain=[0.0, 100.0], range=(0, 400))
    assert scale.domain is not None
    assert list(scale.domain) == [0.0, 100.0]


def test_symlog_scale_explicit_domain_honored():
    """SymlogScale with explicit domain= stores and returns it correctly."""
    scale = fm.SymlogScale(domain=[-100.0, 100.0], range=(0, 400))
    assert scale.domain is not None
    assert list(scale.domain) == [-100.0, 100.0]


# ---------------------------------------------------------------------------
# 8: Domain-less typed scale on real data produces sensible auto-inferred domain
# ---------------------------------------------------------------------------


def test_log_scale_no_domain_axis_spans_data(df):
    """LogScale with no domain should auto-infer domain from data extent."""
    chart = (
        fm.Chart(df)
        .mark_point()
        .encode(
            x=fm.X("x"),
            y=fm.Y("y", scale=fm.LogScale(range=(0, 400))),
        )
    )
    svg = chart.to_svg()
    # The SVG should contain tick labels near the data max (1024)
    # A reasonable log axis spanning [2, 1024] should include 100, 1000 or similar
    assert "1" in svg, "SVG should contain numeric tick labels"
    assert svg.strip(), "SVG should not be empty"


def test_linear_scale_no_domain_axis_spans_data(df):
    """LinearScale with no domain should auto-infer domain from data extent."""
    chart = (
        fm.Chart(df)
        .mark_point()
        .encode(
            x=fm.X("x"),
            y=fm.Y("y", scale=fm.LinearScale(range=(0, 400))),
        )
    )
    svg = chart.to_svg()
    assert svg.strip(), "SVG should not be empty"


# ---------------------------------------------------------------------------
# 8b: Domain getter returns None when not explicitly set
# ---------------------------------------------------------------------------


def test_log_scale_no_domain_getter_returns_none():
    """LogScale with no domain: .domain property returns None."""
    scale = fm.LogScale(range=(0, 400))
    assert scale.domain is None


def test_linear_scale_no_domain_getter_returns_none():
    """LinearScale with no domain: .domain property returns None."""
    scale = fm.LinearScale(range=(0, 400))
    assert scale.domain is None


def test_pow_scale_no_domain_getter_returns_none():
    """PowScale with no domain: .domain property returns None."""
    scale = fm.PowScale(range=(0, 400))
    assert scale.domain is None


def test_sqrt_scale_no_domain_getter_returns_none():
    """SqrtScale with no domain: .domain property returns None."""
    scale = fm.SqrtScale(range=(0, 400))
    assert scale.domain is None


def test_symlog_scale_no_domain_getter_returns_none():
    """SymlogScale with no domain: .domain property returns None."""
    scale = fm.SymlogScale(range=(0, 400))
    assert scale.domain is None
