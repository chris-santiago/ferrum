"""theme.color_scheme drives categorical color assignment.

Precedence: encoding.scheme (per-encoding override) > theme.color_scheme
(Theme default) > OKABE_ITO fallback. Sequential scheme names on nominal
color encodings substitute tableau10 (the canonical Vega-Lite categorical
default) rather than collapsing silently.
"""

from __future__ import annotations

import polars as pl

import ferrum as fm


def _multi_series_chart() -> fm.Chart:
    df = pl.DataFrame(
        {
            "x": [1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0],
            "y": [4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
            "cat": ["a", "a", "a", "b", "b", "b", "c", "c", "c"],
        }
    )
    return fm.Chart(df).mark_point().encode(x="x", y="y", color="cat")


def test_theme_color_scheme_switches_palette() -> None:
    chart = _multi_series_chart()
    svg_tab = chart.theme(fm.Theme(color_scheme="tableau10")).to_svg()
    svg_s1 = chart.theme(fm.Theme(color_scheme="set1")).to_svg()
    assert svg_tab != svg_s1, "tableau10 and set1 must produce different SVGs"
    # tableau10 first color = #4C78A8; set1 first color = #E41A1C.
    assert "#4c78a8" in svg_tab.lower() or "rgb(76, 120, 168)" in svg_tab.lower()
    assert "#e41a1c" in svg_s1.lower() or "rgb(228, 26, 28)" in svg_s1.lower()


def test_palette_wraps_past_length() -> None:
    # tableau10 has 10 colors; 12 categories must wrap, render must succeed.
    df = pl.DataFrame(
        {
            "x": list(range(12)),
            "y": list(range(12)),
            "cat": list("abcdefghijkl"),
        }
    )
    svg = (
        fm.Chart(df)
        .mark_point()
        .encode(x="x", y="y", color="cat")
        .theme(fm.Theme(color_scheme="tableau10"))
        .to_svg()
    )
    assert "<svg" in svg


def test_sequential_scheme_falls_back_to_tableau10_on_nominal() -> None:
    chart = _multi_series_chart()
    svg_viridis = chart.theme(fm.Theme(color_scheme="viridis")).to_svg()
    svg_tab = chart.theme(fm.Theme(color_scheme="tableau10")).to_svg()
    # Sequential scheme on a nominal encoding substitutes tableau10 — so the
    # categorical color path should produce byte-identical SVGs.
    assert svg_viridis == svg_tab


def test_no_theme_equals_explicit_default_scheme() -> None:
    # When no theme is supplied, the renderer uses ThemeInputs::default()
    # which carries color_scheme="paper_ink". Verify that an explicit
    # Theme(color_scheme="paper_ink") produces a byte-identical SVG —
    # confirms the default theme path doesn't drift from the explicit-theme path.
    chart = _multi_series_chart()
    svg_default = chart.to_svg()
    svg_explicit = chart.theme(fm.Theme(color_scheme="paper_ink")).to_svg()
    assert svg_default == svg_explicit
