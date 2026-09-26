"""``mark_boxen()``'s median mark and axis titles.

F-L03-02/12 (frozen texts: ``findings/L03.md``): the median mark used to be a
bare ``rule`` with only a single positional value bound (``x=<category>``,
``y=<lower>``, no ``x2``/``y2``), which ``rule.rs``'s documented shape table
resolves to a full-panel ``HorizontalSpan`` -- the median line ran across
every category, not just its own. Separately, every boxen layer's positional
encoding referenced ``LetterValue``'s internal output column names
(``"group"``, ``"lower"``, ``"upper"``, ``"value"``) directly, so the
rendered axes showed those instead of the user's actual field names.

This module pins both fixes on the real render (SVG regex, not the desugar
IR) so a regression in either the mark-type/``band_size`` change or the
``X(title=...)``/``Y(title=...)`` wiring is caught at the pixel-adjacent
layer, not just the spec-building layer.

Per the repo's Python test-file convention this is feature-scoped coverage
for the boxen median/titling seam, not a findings-ID file.
"""

from __future__ import annotations

import math
import re

import polars as pl

import ferrum as fm


def _species_fixture() -> pl.DataFrame:
    """Three categories, distinct field names from LetterValue's internal
    ``"group"``/``"lower"``/``"upper"``/``"value"`` column names, so a title
    leak is unambiguous rather than a coincidental match."""
    return pl.DataFrame(
        {
            "species": ["setosa"] * 40 + ["versicolor"] * 40 + ["virginica"] * 40,
            "petal_length": (
                [float(i) for i in range(40)]
                + [float(i) for i in range(20, 60)]
                + [float(i) for i in range(5, 45)]
            ),
        }
    )


def _axis_titles(svg: str) -> set[str]:
    """Every axis-title ``<text>`` element's content. Boxen's theme renders
    axis titles at ``font-weight="600"``, distinct from tick labels
    (unweighted) -- both share the same font-family/fill otherwise."""
    return set(re.findall(r'<text[^>]*font-weight="600"[^>]*>([^<]*)</text>', svg))


def _horizontal_line_widths(svg: str) -> list[float]:
    """The (x2 - x1) width of every horizontal (``y1 == y2``) ``<line>``."""
    lines = re.findall(r'<line x1="([-\d.]+)" y1="([-\d.]+)" x2="([-\d.]+)" y2="([-\d.]+)"', svg)
    return [
        round(abs(float(x2) - float(x1)), 3)
        for x1, y1, x2, y2 in lines
        if abs(float(y1) - float(y2)) < 1e-9
    ]


def _depth_band_rect_widths(svg: str) -> list[float]:
    """The width of every depth-band ``<rect>`` (identified by its
    ``rgba(...)`` opacity-ramp fill, which only depth bands use under the
    default ``palette=None``)."""
    rects = re.findall(
        r'<rect x="[-\d.]+" y="[-\d.]+" width="([-\d.]+)" height="[-\d.]+" fill="rgba', svg
    )
    return [round(float(w), 3) for w in rects]


def test_boxen_axis_titles_show_user_field_names():
    """Both axes show the user's own field names, not LetterValue's
    internal ``"group"``/``"lower"``/``"upper"``/``"value"`` column names."""
    svg = fm.Chart(_species_fixture()).mark_boxen().encode(x="species", y="petal_length").to_svg()
    titles = _axis_titles(svg)
    assert "species" in titles
    assert "petal_length" in titles
    assert titles.isdisjoint({"group", "lower", "upper", "value"}), (
        f"LetterValue's internal column names leaked onto the axes: {titles}"
    )


def test_boxen_axis_titles_show_user_field_names_horizontal():
    """Same pin, ``horizontal=True`` (x/y swapped internally)."""
    svg = (
        fm.Chart(_species_fixture())
        .mark_boxen(horizontal=True)
        .encode(x="petal_length", y="species")
        .to_svg()
    )
    titles = _axis_titles(svg)
    assert "species" in titles
    assert "petal_length" in titles
    assert titles.isdisjoint({"group", "lower", "upper", "value"})


def test_boxen_median_tick_span_confined_to_its_category_band():
    """The median mark's rendered span matches a real depth band's own
    width (one category's band), strictly less than the full panel width
    -- proving it no longer draws a rule spanning every category
    (F-L03-02)."""
    svg = fm.Chart(_species_fixture()).mark_boxen().encode(x="species", y="petal_length").to_svg()
    widths = _horizontal_line_widths(svg)
    panel_width = max(widths)
    band_widths = _depth_band_rect_widths(svg)
    assert band_widths, "expected at least one depth-band rect to compare against"
    band_width = band_widths[0]
    assert not math.isclose(band_width, panel_width), (
        "fixture must have >1 category so band width != panel width"
    )

    median_widths = [w for w in widths if math.isclose(w, band_width, rel_tol=1e-6)]
    assert len(median_widths) == 3, (
        f"expected one band-width median tick per category (3), got widths {widths}"
    )
    for w in median_widths:
        assert w < panel_width, (
            f"median tick width {w} should be confined to one category's band, "
            f"not the full panel width {panel_width}"
        )


def test_boxen_median_tick_span_confined_to_its_category_band_horizontal():
    """Same confinement pin as the vertical-orientation test above, for
    ``horizontal=True``: the median tick takes a different Rust branch
    there (``tick.rs``'s "ordinal y + quantitative x" vertical-tick mode,
    a strip-plot-shaped segment rather than the vertical case's horizontal
    one), so it needs its own render-level check rather than trusting the
    vertical pin to cover both."""
    svg = (
        fm.Chart(_species_fixture())
        .mark_boxen(horizontal=True)
        .encode(x="petal_length", y="species")
        .to_svg()
    )
    lines = re.findall(r'<line x1="([-\d.]+)" y1="([-\d.]+)" x2="([-\d.]+)" y2="([-\d.]+)"', svg)
    vertical_heights = [
        round(abs(float(y2) - float(y1)), 3)
        for x1, y1, x2, y2 in lines
        if abs(float(x1) - float(x2)) < 1e-9
    ]
    panel_height = max(vertical_heights)
    band_heights = [
        round(float(h), 3)
        for h in re.findall(
            r'<rect x="[-\d.]+" y="[-\d.]+" width="[-\d.]+" height="([-\d.]+)" fill="rgba', svg
        )
    ]
    assert band_heights, "expected at least one depth-band rect to compare against"
    band_height = band_heights[0]
    assert not math.isclose(band_height, panel_height), (
        "fixture must have >1 category so band height != panel height"
    )

    median_heights = [h for h in vertical_heights if math.isclose(h, band_height, abs_tol=0.01)]
    assert len(median_heights) == 3, (
        f"expected one band-height median tick per category (3), got heights {vertical_heights}"
    )
    for h in median_heights:
        assert h < panel_height, (
            f"median tick height {h} should be confined to one category's band, "
            f"not the full panel height {panel_height}"
        )


def test_boxen_median_declares_both_positional_channels():
    """The median layer's encoding carries both the category and value
    channels (not just one) -- a single-channel ``tick`` normalizes to a
    rug, which the median mark must never become."""
    from ferrum.marks.composite import desugar_boxen

    result = desugar_boxen("species", "petal_length")
    median = next(layer for layer in result.layers if layer.name == "median")
    assert median.mark == "tick"
    assert "x" in median.encoding and "y" in median.encoding


def test_boxen_default_opacity_ramp_decreases_with_depth_in_paint_order():
    """Under the default ``palette=None`` shading, bands paint in ascending
    depth order and the opacity ramp strictly decreases with depth (outer
    bands are more opaque, matching the mark's documented outer->inner
    ramp) -- a direction-shaped pin on the ramp itself, independent of any
    specific opacity value."""
    values = [float(i) for i in range(200)]
    df = pl.DataFrame({"g": ["a"] * len(values), "v": values})
    svg = fm.Chart(df).mark_boxen().encode(x="g", y="v").to_svg()
    opacities = [float(a) for a in re.findall(r'fill="rgba\([^)]*,\s*([\d.]+)\)"', svg)]
    assert len(opacities) >= 3, "need multiple real depth bands to prove a ramp direction"
    assert opacities == sorted(opacities, reverse=True) and len(set(opacities)) == len(opacities), (
        f"opacity ramp is not strictly decreasing in paint (depth) order: {opacities}"
    )


def test_boxen_opacity_ramp_step_normalizes_by_real_materialized_depth():
    """Pins the ramp's DENOMINATOR semantics, not just its direction
    (quality-review REVISE, 2026-09-06 cycle 2, finding 3/6): the direction
    test above is green whether the ramp is normalized by the chart's real
    materialized depth or by the mark's full configured cap -- both are
    still strictly decreasing. The adjacent-band opacity STEP is the tell:
    it must equal ``0.55 / (real_depth - 1)``, where ``real_depth`` is this
    fixture's own Tukey depth (independently computed via
    ``composite._k_for_depth``, the exact formula ``_boxen_materialized_depth``
    injects), not ``0.55 / (depth_count - 1)`` (the full registry-derived
    cap, currently 12) -- the two denominators differ by roughly 4x for
    this fixture, so this discriminates cleanly."""
    from ferrum.marks.composite import _boxen_depth_count, _k_for_depth

    n = 200
    values = [float(i) for i in range(n)]
    df = pl.DataFrame({"g": ["a"] * n, "v": values})

    real_depth = _k_for_depth("tukey", 0.007, n)
    depth_count = _boxen_depth_count()
    assert 2 <= real_depth < depth_count, (
        "fixture must materialize a real depth strictly between the base "
        "band and the full configured cap, or this test can't discriminate"
    )
    expected_step = 0.55 / (real_depth - 1)

    svg = fm.Chart(df).mark_boxen().encode(x="g", y="v").to_svg()
    opacities = [float(a) for a in re.findall(r'fill="rgba\([^)]*,\s*([\d.]+)\)"', svg)]
    assert len(opacities) == real_depth, (
        f"expected exactly {real_depth} real bands for this fixture, got {len(opacities)}"
    )
    steps = [round(a - b, 4) for a, b in zip(opacities, opacities[1:])]
    assert all(math.isclose(step, expected_step, abs_tol=2e-3) for step in steps), (
        f"opacity steps {steps} do not match the real-depth-normalized step "
        f"{expected_step:.4f} (real_depth={real_depth}) -- the ramp is "
        f"normalizing against the configured cap ({depth_count}) instead of "
        f"this chart's real materialized depth"
    )
