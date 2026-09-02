"""Python-level SVG pins for the ``stroke_dash`` channel's rendered output.

Batch-A T12 spec review (2026-09-01) flagged that the audit's reported
symptom surface — ``fm.relplot(style=<categorical>, kind="line")`` rendering
one polyline per style category with a distinct dash pattern and a dash
legend — had coverage only at the Rust orchestration-test level (hand-built
specs), with no Python-level regression pin exercising the real
``fm.relplot`` -> ``Chart`` -> ``to_svg()`` path. This module closes that gap,
and additionally pins that ``StrokeDash(sort=...)`` (batch-A T12's other
finding: ``sort=`` was silently dropped before moving to
``APPEARANCE_SORT`` — see ``tests/test_appearance_honored_kwargs.py``)
actually reorders which category gets which dash pattern, not just that the
kwarg is accepted.
"""

from __future__ import annotations

import re

import polars as pl

import ferrum as fm


def _style_df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "x": [1.0, 2.0, 3.0, 4.0] * 3,
            "y": [1.0, 2.0, 1.5, 3.0, 2.0, 2.5, 3.5, 4.0, 0.5, 1.0, 1.2, 1.8],
            "style_col": ["a"] * 4 + ["b"] * 4 + ["c"] * 4,
        }
    )


def _legend_dash_pairs(svg: str, title: str) -> list[tuple[str, str | None]]:
    """Return ``(label, dasharray_or_None)`` pairs for the legend titled *title*.

    The dash legend renders a ``<line>`` swatch immediately followed by a
    ``<text>`` label for each domain value, in domain order, right after the
    legend's title ``<text>``. This walks that run directly off the raw SVG
    (rather than a generic ElementTree tree-walk) because pairing swatch to
    label relies on their emission adjacency, not tree structure.
    """
    idx = svg.index(f">{title}<")
    tail = svg[idx:]
    raw_pairs = re.findall(r"<line([^>]*)/><text[^>]*>([^<]*)</text>", tail)
    pairs: list[tuple[str, str | None]] = []
    for attrs, label in raw_pairs:
        m = re.search(r'stroke-dasharray="([^"]+)"', attrs)
        pairs.append((label, m.group(1) if m else None))
    return pairs


def _polyline_dashes(svg: str) -> list[str | None]:
    """Return the ``stroke-dasharray`` (or ``None`` for solid) of each ``<polyline>``."""
    out: list[str | None] = []
    for p in re.findall(r"<polyline[^>]*/>", svg):
        m = re.search(r'stroke-dasharray="([^"]+)"', p)
        out.append(m.group(1) if m else None)
    return out


# ---------------------------------------------------------------------------
# relplot(style=, kind="line") end-to-end: n polylines, n distinct dash
# patterns, dash legend with one entry per category.
# ---------------------------------------------------------------------------


def test_relplot_style_line_renders_one_polyline_per_category():
    chart = fm.relplot(_style_df(), x="x", y="y", style="style_col", kind="line")
    svg = chart.to_svg()

    assert svg.count("<polyline") == 3, (
        f"expected 3 polylines (one per style_col category); got:\n{svg[:2000]}"
    )


def test_relplot_style_line_polylines_have_distinct_dash_patterns():
    chart = fm.relplot(_style_df(), x="x", y="y", style="style_col", kind="line")
    svg = chart.to_svg()

    dashes = _polyline_dashes(svg)
    assert len(dashes) == 3
    assert len(set(dashes)) == 3, f"expected 3 distinct dash patterns, got {dashes}"

    # Discriminating: pin that the mark polylines and the legend swatches
    # agree on the *set* of dash patterns used, not just that each is
    # independently distinct — an implementation that assigned mark dashes
    # from one domain order and legend swatches from another would still
    # pass the two assertions above.
    pairs = _legend_dash_pairs(svg, "style_col")
    assert set(dashes) == {d for _, d in pairs}, (
        f"mark polyline dashes {set(dashes)} should match legend dashes {{d for _, d in pairs}}"
    )


def test_relplot_style_line_renders_dash_legend_with_category_labels():
    chart = fm.relplot(_style_df(), x="x", y="y", style="style_col", kind="line")
    svg = chart.to_svg()

    pairs = _legend_dash_pairs(svg, "style_col")
    assert [label for label, _ in pairs] == ["a", "b", "c"], (
        f"expected dash legend entries a, b, c in domain order; got {pairs}"
    )
    # Domain order maps: index 0 = solid, then a distinct dash pattern per
    # subsequent category (mirrors the numeric stroke_dash index mapping
    # pinned in tests/test_silent_drop_remediation.py::TestStrokeDashSVG).
    labels_to_dash = dict(pairs)
    assert labels_to_dash["a"] is None, "first domain category should be solid"
    assert labels_to_dash["b"] is not None
    assert labels_to_dash["c"] is not None
    assert labels_to_dash["b"] != labels_to_dash["c"]


# ---------------------------------------------------------------------------
# StrokeDash(sort=...) reorders the dash assignment (discriminating: which
# category gets which dasharray shifts, not just that sort= is accepted).
# ---------------------------------------------------------------------------


def test_stroke_dash_sort_reorders_dash_assignment():
    df = _style_df()

    default_svg = (
        fm.Chart(df)
        .mark_line()
        .encode(x="x", y="y", stroke_dash=fm.StrokeDash("style_col"))
        .to_svg()
    )
    sorted_svg = (
        fm.Chart(df)
        .mark_line()
        .encode(x="x", y="y", stroke_dash=fm.StrokeDash("style_col", sort=["c", "b", "a"]))
        .to_svg()
    )

    default_pairs = _legend_dash_pairs(default_svg, "style_col")
    sorted_pairs = _legend_dash_pairs(sorted_svg, "style_col")

    assert [label for label, _ in default_pairs] == ["a", "b", "c"]
    assert [label for label, _ in sorted_pairs] == ["c", "b", "a"], (
        f"sort=['c','b','a'] should reorder the legend domain; got {sorted_pairs}"
    )

    # Discriminating: 'a' moves from solid (index 0) under the default
    # ascending domain to the last (dotted) pattern under the explicit
    # reversed sort — not merely relabeled, the dash assignment itself shifts.
    default_dash = dict(default_pairs)
    sorted_dash = dict(sorted_pairs)
    assert default_dash["a"] is None
    assert sorted_dash["a"] is not None
    assert default_dash["a"] != sorted_dash["a"]
    assert default_dash["c"] != sorted_dash["c"]
