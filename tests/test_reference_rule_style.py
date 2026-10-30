"""Pinning test for the reference-rule style on a `+` span rule.

A span rule (`mark_rule` binding exactly one positional channel, no `x2`/
`y2`) composed with `+` beside a colored host already draws its own default
style -- gray `#9ca3af`, dash `4,4`, width `1.5` -- because a `+` operand's
bindings are scoped to that operand's own layers (sibling isolation): the
rule layer never receives the host's `color` binding in the first place, so
there is nothing to clear. This test pins that: the span rule draws its own
reference style, and the host's color legend survives untouched. It exists
only so a future change to span-rule defaults or to `+`'s sibling isolation
shows up as a failure here.
"""

from __future__ import annotations

import polars as pl

import ferrum as fm
from tests._svg_rects import svg_elements


def test_scatter_plus_rule_draws_reference_style_and_keeps_the_scatter_legend() -> None:
    df = pl.DataFrame(
        {
            "x": [1.0, 2.0, 3.0, 4.0],
            "y": [1.0, 2.0, 3.0, 4.0],
            "class": ["Alpha", "Beta", "Gamma", "Alpha"],
            "ref_y": [2.5, 2.5, 2.5, 2.5],
        }
    )
    chart = fm.Chart(df).mark_point().encode(x="x", y="y", color="class") + fm.Chart(
        df
    ).mark_rule().encode(y="ref_y")
    svg = chart.to_svg()

    rule_lines = svg_elements(svg, "line", lambda a: a.get("stroke") == "#9ca3af")
    assert len(rule_lines) == 4, "one span-rule line per row of the rule's own data"
    for line in rule_lines:
        assert line["stroke-width"] == "1.5"
        assert line["stroke-dasharray"] == "4,4"
        # No `opacity`/`stroke-opacity` attribute at all: the rule draws its
        # own constant, full-opacity style, never the scatter's per-row
        # opacity encoding (there is none here, but the attributes' absence
        # pins that the rule's style is not computed per row).
        assert "opacity" not in line
        assert "stroke-opacity" not in line

    # The scatter's own color legend survives untouched: title plus every
    # category.
    assert ">class<" in svg
    for category in ("Alpha", "Beta", "Gamma"):
        assert f">{category}<" in svg
