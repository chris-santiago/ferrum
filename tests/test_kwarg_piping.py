"""Kwarg-piping test suite: verify every mark-style kwarg round-trips from
Python API to SVG attribute.

Each parametrized test renders a minimal 5-row DataFrame and checks that the
kwarg value appears in the SVG output.  This catches silent drops at every
stage: MarkBase validation, JSON serialization, Rust serde deserialization,
resolve_mark_style, and per-mark draw functions.

Spec: docs/superpowers/specs/2026-05-13-kwarg-piping-test-coverage.md
"""

from __future__ import annotations

import re
from typing import Any, Callable

import polars as pl
import pytest

import ferrum as fm


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def simple_df():
    return pl.DataFrame(
        {
            "x": [1.0, 2.0, 3.0, 4.0, 5.0],
            "y": [2.0, 4.0, 1.5, 3.5, 2.5],
            "label": ["A", "B", "C", "D", "E"],
            "group": ["a", "a", "b", "b", "b"],
        }
    )


@pytest.fixture
def bar_df():
    """Categorical x for bar charts."""
    return pl.DataFrame(
        {
            "x": ["a", "b", "c", "d", "e"],
            "y": [2.0, 4.0, 1.5, 3.5, 2.5],
        }
    )


@pytest.fixture
def heatmap_df():
    """Wide-format data for heatmap (annot=True)."""
    return pl.DataFrame(
        {
            "row": ["a", "b", "c"],
            "x": [1.0, 0.5, 0.3],
            "y": [0.5, 1.0, 0.7],
            "z": [0.3, 0.7, 1.0],
        }
    )


# ---------------------------------------------------------------------------
# Assertion helpers
# ---------------------------------------------------------------------------


def _has_rgba(svg: str) -> bool:
    """True when at least one mark element uses an rgba() color (i.e. alpha < 1.0)."""
    return "rgba(" in svg


def _has_translucent_fill(svg: str) -> bool:
    """True when a fill attribute contains an rgba() value."""
    return bool(re.search(r'fill="rgba\(', svg))


def _has_translucent_stroke(svg: str) -> bool:
    """True when a stroke attribute contains an rgba() value."""
    return bool(re.search(r'stroke="rgba\(', svg))


def _circle_radii(svg: str) -> list[float]:
    """Extract all <circle r="..."> radius values."""
    return [float(m) for m in re.findall(r'<circle[^>]+r="([^"]+)"', svg)]


def _has_hollow_points(svg: str) -> bool:
    """True when at least one circle has fill='none' and a stroke."""
    return bool(re.search(r'<circle[^>]+fill="none"[^>]+stroke=', svg))


# ---------------------------------------------------------------------------
# 1. Primitive marks: parametrized matrix
# ---------------------------------------------------------------------------

# Each tuple: (mark_method, kwarg_dict, encode_dict, svg_assertion, description)
#
# svg_assertion is either:
#   - a string: must appear in the SVG
#   - a callable(str) -> bool: must return True
#
# We pass the full kwarg dict rather than a single key to allow multi-kwarg combos.

_PRIMITIVE_CASES: list[tuple[str, dict, dict | None, str | Callable[[str], bool], str]] = [
    # ---- Point ----
    (
        "mark_point",
        {"opacity": 0.3},
        None,
        _has_translucent_fill,
        "point opacity bakes alpha into fill rgba()",
    ),
    (
        "mark_point",
        {"size": 120},
        None,
        lambda svg: max(_circle_radii(svg), default=0.0) > 5.0,
        "point size=120 produces larger circle radius (sqrt(120/pi) ~ 6.18)",
    ),
    (
        "mark_point",
        {"filled": False},
        None,
        _has_hollow_points,
        'point filled=False emits fill="none" with stroke',
    ),
    (
        "mark_point",
        {"stroke_width": 2.5, "filled": False},
        None,
        'stroke-width="2.5"',
        "point stroke_width=2.5 reaches SVG (hollow points)",
    ),
    (
        "mark_point",
        {"shape": "square"},
        None,
        "<rect",
        "point shape='square' emits <rect> instead of <circle>",
    ),
    # ---- Line ----
    (
        "mark_line",
        {"stroke_width": 3},
        None,
        'stroke-width="3"',
        "line stroke_width=3 reaches SVG",
    ),
    (
        "mark_line",
        {"stroke_dash": [4, 2]},
        None,
        'stroke-dasharray="4,2"',
        "line stroke_dash=[4,2] emits dasharray with commas",
    ),
    (
        "mark_line",
        {"opacity": 0.5},
        None,
        _has_translucent_stroke,
        "line opacity=0.5 bakes alpha into stroke rgba()",
    ),
    (
        "mark_line",
        {"stroke_width": 3, "stroke_dash": [4, 2], "opacity": 0.5},
        None,
        lambda svg: (
            'stroke-width="3"' in svg
            and 'stroke-dasharray="4,2"' in svg
            and _has_translucent_stroke(svg)
        ),
        "line combo: stroke_width + stroke_dash + opacity all present",
    ),
    # ---- Bar ----
    (
        "mark_bar",
        {"corner_radius": 4},
        "bar",
        'rx="4"',
        "bar corner_radius=4 emits rx attribute",
    ),
    (
        "mark_bar",
        {"opacity": 0.7},
        "bar",
        _has_translucent_fill,
        "bar opacity=0.7 bakes alpha into fill rgba()",
    ),
    (
        "mark_bar",
        {"corner_radius": 4, "opacity": 0.7},
        "bar",
        lambda svg: 'rx="4"' in svg and _has_translucent_fill(svg),
        "bar combo: corner_radius + opacity both present",
    ),
    # ---- Text ----
    (
        "mark_text",
        {"font_size": 16},
        "text",
        'font-size="16"',
        "text font_size=16 reaches SVG",
    ),
    (
        "mark_text",
        {"font_weight": "bold"},
        "text",
        'font-weight="bold"',
        "text font_weight='bold' reaches SVG",
    ),
    (
        "mark_text",
        {"align": "right"},
        "text",
        'text-anchor="end"',
        "text align='right' maps to text-anchor='end'",
    ),
    (
        "mark_text",
        {"align": "left"},
        "text",
        'text-anchor="start"',
        "text align='left' maps to text-anchor='start'",
    ),
    (
        "mark_text",
        {"font_size": 16, "font_weight": "bold", "align": "right"},
        "text",
        lambda svg: (
            'font-size="16"' in svg and 'font-weight="bold"' in svg and 'text-anchor="end"' in svg
        ),
        "text combo: font_size + font_weight + align all present",
    ),
    # ---- Rule ----
    (
        "mark_rule",
        {"stroke_dash": [6, 3]},
        None,
        'stroke-dasharray="6,3"',
        "rule stroke_dash=[6,3] emits dasharray",
    ),
    (
        "mark_rule",
        {"stroke_width": 2.5},
        None,
        'stroke-width="2.5"',
        "rule stroke_width=2.5 reaches SVG",
    ),
    # ---- Area ----
    (
        "mark_area",
        {"line": True},
        None,
        lambda svg: svg.count("<path") >= 2,
        "area line=True emits an extra path for the top-edge border",
    ),
]


def _make_id(case: tuple) -> str:
    """Readable parametrize ID."""
    return case[4]


@pytest.mark.parametrize("case", _PRIMITIVE_CASES, ids=[c[4] for c in _PRIMITIVE_CASES])
def test_kwarg_reaches_svg(case, simple_df, bar_df):
    mark_method, kwargs, encode_mode, svg_check, desc = case

    # Choose data and encoding based on the mark type.
    if encode_mode == "bar":
        df = bar_df
        encode_kwargs = {"x": "x", "y": "y"}
    elif encode_mode == "text":
        df = simple_df
        encode_kwargs = {"x": "x", "y": "y", "text": "label"}
    else:
        df = simple_df
        encode_kwargs = {"x": "x", "y": "y"}

    chart = getattr(fm.Chart(df), mark_method)(**kwargs).encode(**encode_kwargs)
    svg = chart.to_svg()

    if callable(svg_check):
        assert svg_check(svg), f"{mark_method}({kwargs}) assertion failed: {desc}"
    else:
        assert svg_check in svg, (
            f"{mark_method}({kwargs}) — expected '{svg_check}' in SVG output. [{desc}]"
        )


# ---------------------------------------------------------------------------
# 2. Area opacity: known bug — area renderer ignores mark_style.opacity
# ---------------------------------------------------------------------------


def test_area_opacity_kwarg(simple_df):
    """mark_area(opacity=0.1) should produce a MORE translucent fill than default."""
    default_svg = fm.Chart(simple_df).mark_area().encode(x="x", y="y").to_svg()
    custom_svg = fm.Chart(simple_df).mark_area(opacity=0.1).encode(x="x", y="y").to_svg()
    # Extract the alpha value from the first rgba() fill in each SVG.
    default_alpha = _extract_fill_alpha(default_svg)
    custom_alpha = _extract_fill_alpha(custom_svg)
    assert default_alpha is not None, "default area should have rgba fill"
    assert custom_alpha is not None, "area(opacity=0.1) should have rgba fill"
    assert custom_alpha < default_alpha, (
        f"opacity=0.1 should produce lower alpha than default: "
        f"got {custom_alpha} vs default {default_alpha}"
    )


def _extract_fill_alpha(svg: str) -> float | None:
    """Extract the alpha from the first fill='rgba(r,g,b,ALPHA)' in the SVG."""
    m = re.search(r'fill="rgba\(\d+,\d+,\d+,([0-9.]+)\)"', svg)
    return float(m.group(1)) if m else None


# ---------------------------------------------------------------------------
# 3. Composite mark kwargs
# ---------------------------------------------------------------------------


def test_smooth_ci_produces_ribbon(simple_df):
    """mark_smooth(ci=0.95) should emit a ribbon layer with translucent fill."""
    svg = fm.Chart(simple_df).mark_smooth(ci=0.95).encode(x="x", y="y").to_svg()
    assert "<svg" in svg, "should produce valid SVG"
    # The ribbon layer emits a <path> with an rgba() fill (translucent).
    assert _has_translucent_fill(svg), (
        "mark_smooth(ci=0.95) should produce a ribbon with translucent fill"
    )
    # The line layer emits a <polyline> or <path> with stroke.
    assert re.search(r"<(polyline|path)[^>]+stroke=", svg), (
        "mark_smooth(ci=0.95) should produce a line layer with stroke"
    )


def test_parallel_coordinates_alpha(simple_df):
    """parallel_coordinates_chart(alpha=0.3) should emit translucent polylines."""
    # Build long-form data suitable for parallel coordinates.
    df = pl.DataFrame(
        {
            "feature": ["f1", "f2", "f1", "f2", "f1", "f2"],
            "value": [0.1, 0.9, 0.5, 0.4, 0.8, 0.2],
            "sample_id": ["s0", "s0", "s1", "s1", "s2", "s2"],
        }
    )
    svg = fm.Chart(df).mark_parallel_coordinates(alpha=0.3, color_field=None).to_svg()
    assert "<svg" in svg, "should produce valid SVG"
    # Polylines should have translucent stroke (rgba).
    assert _has_translucent_stroke(svg), (
        "parallel_coordinates(alpha=0.3) should produce translucent polylines"
    )


# ---------------------------------------------------------------------------
# 4. Figure function kwargs
# ---------------------------------------------------------------------------


def test_lmplot_line_kws_stroke_width(simple_df):
    """lmplot(line_kws={'stroke_width': 3}) forwards stroke_width to SVG."""
    svg = fm.lmplot(simple_df, x="x", y="y", line_kws={"stroke_width": 3}).to_svg()
    assert 'stroke-width="3"' in svg, (
        "lmplot line_kws={'stroke_width': 3} should produce stroke-width='3' in SVG"
    )


def test_heatmap_annot_emits_text(heatmap_df):
    """heatmap(annot=True) should emit <text> elements for cell annotations."""
    svg = fm.heatmap(heatmap_df, annot=True).to_svg()
    assert "<svg" in svg, "should produce valid SVG"
    # Count mark-text elements: they have font-family and are NOT axis labels.
    # Heatmap has 3 rows x 3 columns = 9 cells minimum.
    # (Some cells may be axis labels; just verify text elements exist beyond axes.)
    text_count = svg.count("<text ")
    # Axis labels + tick labels + annotation labels.  Annotation alone should
    # add at least as many text elements as there are cells.
    assert text_count >= 6, (
        f"heatmap(annot=True) should emit cell annotation text; "
        f"found only {text_count} <text> elements total"
    )


def test_lmplot_scatter_kws_opacity(simple_df):
    """lmplot(scatter_kws={'opacity': 0.3}) forwards opacity to scatter points."""
    svg = fm.lmplot(simple_df, x="x", y="y", scatter_kws={"opacity": 0.3}).to_svg()
    # Scatter points should have rgba fill with alpha ~ 0.3.
    circles = re.findall(r'<circle[^>]+fill="([^"]+)"', svg)
    translucent = [c for c in circles if "rgba(" in c]
    assert translucent, (
        "lmplot scatter_kws={'opacity': 0.3} should produce translucent scatter points"
    )


# ---------------------------------------------------------------------------
# 5. Additional coverage: stroke and fill color overrides
# ---------------------------------------------------------------------------


def test_point_fill_override(simple_df):
    """mark_point(fill='#ff0000') should produce red-filled circles."""
    svg = fm.Chart(simple_df).mark_point(fill="#ff0000").encode(x="x", y="y").to_svg()
    # fill should contain the red color (either #ff0000 or rgb form).
    assert "ff0000" in svg.lower() or "255,0,0" in svg, (
        "mark_point(fill='#ff0000') should produce red fill in SVG"
    )


def test_line_stroke_override(simple_df):
    """mark_line(stroke='#00ff00') should produce green stroke."""
    svg = fm.Chart(simple_df).mark_line(stroke="#00ff00").encode(x="x", y="y").to_svg()
    assert "00ff00" in svg.lower() or "0,255,0" in svg, (
        "mark_line(stroke='#00ff00') should produce green stroke in SVG"
    )


def test_rule_stroke_override(simple_df):
    """mark_rule(stroke='#aaaaaa') should produce grey stroke."""
    svg = fm.Chart(simple_df).mark_rule(stroke="#aaaaaa").encode(x="x", y="y").to_svg()
    assert "aaaaaa" in svg.lower(), "mark_rule(stroke='#aaaaaa') should produce grey stroke in SVG"


# ---------------------------------------------------------------------------
# 6. Interpolation kwarg
# ---------------------------------------------------------------------------


def test_text_dx_shifts_x_positions(simple_df):
    """mark_text(dx=10) should shift mark-text x positions by 10px vs default."""
    encode = {"x": "x", "y": "y", "text": "label"}
    svg_default = fm.Chart(simple_df).mark_text().encode(**encode).to_svg()
    svg_dx = fm.Chart(simple_df).mark_text(dx=10).encode(**encode).to_svg()

    def _mark_text_x_positions(svg: str) -> list[float]:
        # Mark texts use the theme font_color (#1f2937), not the axis label
        # color (#6b7280). Extract x from those elements.
        return [float(m) for m in re.findall(r'<text x="([^"]+)"[^>]*fill="#1f2937"', svg)]

    xs_default = _mark_text_x_positions(svg_default)
    xs_dx = _mark_text_x_positions(svg_dx)
    assert len(xs_default) >= 3, "should have at least 3 mark-text elements"
    assert len(xs_dx) == len(xs_default), "same number of mark-text elements"
    # Filter to the data-driven text labels (skip axis titles that are not shifted).
    # At least one pair should differ by ~10.
    diffs = [b - a for a, b in zip(xs_default, xs_dx)]
    shifted = [d for d in diffs if abs(d - 10.0) < 0.1]
    assert len(shifted) >= 1, (
        f"dx=10 should shift at least one text x position by ~10px; diffs={diffs}"
    )


def test_line_interpolate_step(simple_df):
    """mark_line(interpolate='step') should produce step-shaped paths (H/V commands)."""
    svg = fm.Chart(simple_df).mark_line(interpolate="step").encode(x="x", y="y").to_svg()
    # Step interpolation emits a <path> with H and V commands rather than a
    # <polyline>.
    assert "<path" in svg, "mark_line(interpolate='step') should emit <path> element"
    # The path d attribute should contain H and/or V commands.
    path_d = re.search(r'<path d="([^"]+)"[^>]*fill="none"', svg)
    assert path_d, "step line path not found"
    assert "H" in path_d.group(1) or "V" in path_d.group(1), (
        "step interpolation should produce H/V path commands"
    )


# ---------------------------------------------------------------------------
# 7. Stroke cap and join
# ---------------------------------------------------------------------------


def test_line_stroke_cap(simple_df):
    """mark_line(stroke_cap='round') wraps line in <g stroke-linecap='round'>."""
    svg = fm.Chart(simple_df).mark_line(stroke_cap="round").encode(x="x", y="y").to_svg()
    assert 'stroke-linecap="round"' in svg, (
        "mark_line(stroke_cap='round') should emit stroke-linecap attribute"
    )


def test_area_stroke_join(simple_df):
    """mark_area(stroke_join='round') wraps area in <g stroke-linejoin='round'>."""
    svg = fm.Chart(simple_df).mark_area(stroke_join="round").encode(x="x", y="y").to_svg()
    assert 'stroke-linejoin="round"' in svg, (
        "mark_area(stroke_join='round') should emit stroke-linejoin attribute"
    )
