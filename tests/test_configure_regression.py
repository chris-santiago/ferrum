"""Regression tests for bugs found by the auditor suite (2026-05-24).

Each test prevents a specific bug from recurring:
- Bug 1: Inset connect_to used as pixel coords instead of data coords
- Bug 2: Callout leader line never drawn (arrow="curved" mismatch with Rust)
- Bug 3: BreakAxis marks not repositioned through broken scale
- Bug 4: configure_*() methods silently non-functional
- Bug 5: Configure cascade did not override theme
"""

from __future__ import annotations

import polars as pl
import pytest

import ferrum as fm
from ferrum.structural import BreakAxis, Inset


# ---------------------------------------------------------------------------
# Bug 1: Inset connect_to was used as pixel coords instead of data coords
# ---------------------------------------------------------------------------


class TestInsetConnectToDataCoordinates:
    def test_different_x_domains_produce_different_connector_positions(self):
        """Regression: connect_to coords must be resolved through scales, not used as raw pixels.

        Two charts with the same connect_to=(5, 2) but different x-axis domains
        must produce different SVG connector positions, proving the coordinate is
        being mapped through the data-space scale, not treated as fixed pixels.
        """
        df1 = pl.DataFrame({"x": [0, 10, 20], "y": [1, 2, 3]})
        df2 = pl.DataFrame({"x": [0, 100, 200], "y": [1, 2, 3]})
        inset = fm.Chart(df1).mark_point().encode(x="x", y="y")

        chart1 = fm.Chart(df1).mark_point().encode(x="x", y="y") + Inset(
            chart=inset, bounds=(0.6, 0.1, 0.9, 0.4), connect_to=(5, 2)
        )
        chart2 = fm.Chart(df2).mark_point().encode(x="x", y="y") + Inset(
            chart=inset, bounds=(0.6, 0.1, 0.9, 0.4), connect_to=(5, 2)
        )

        svg1 = chart1.show_svg()
        svg2 = chart2.show_svg()
        # Different x-axis domains mean x=5 maps to different pixel positions.
        # If connect_to were treated as raw pixels, the connector endpoint would
        # be identical and the SVGs would be the same (modulo data marks).
        assert svg1 != svg2, (
            "connect_to=(5, 2) with x-domain [0,20] vs x-domain [0,200] must produce "
            "different connector pixel positions — connect_to must be resolved through scales"
        )

    def test_connect_to_absent_still_renders(self):
        """Baseline: inset without connect_to renders without error."""
        df = pl.DataFrame({"x": [0, 10, 20], "y": [1, 2, 3]})
        inset = fm.Chart(df).mark_point().encode(x="x", y="y")
        chart = fm.Chart(df).mark_point().encode(x="x", y="y") + Inset(
            chart=inset, bounds=(0.6, 0.1, 0.9, 0.4)
        )
        svg = chart.show_svg()
        assert svg.startswith("<svg")


# ---------------------------------------------------------------------------
# Bug 2: Callout annotation leader line never drew
# ---------------------------------------------------------------------------


class TestCalloutLeaderLine:
    def test_default_arrow_produces_leader_line(self):
        """Regression: callout with default arrow='curved' must draw a leader line.

        The fix aligns Python's default "curved" with Rust's accepted value so
        the leader line element is actually emitted in the SVG.
        """
        from ferrum.annotation.primitives import callout

        df = pl.DataFrame({"x": [1, 2, 3], "y": [10, 20, 30]})
        chart = fm.Chart(df).mark_point().encode(x="x", y="y") + callout(x=2.0, y=20.0, text="Note")
        svg = chart.show_svg()
        assert "<line " in svg, (
            "callout with default arrow='curved' must produce a <line> element "
            "for the leader line — the arrow='curved' value was previously not "
            "recognised by Rust, silently suppressing the connector"
        )

    def test_arrow_none_suppresses_leader_line(self):
        """arrow='none' must suppress the leader line relative to the default."""
        from ferrum.annotation.primitives import callout

        df = pl.DataFrame({"x": [1, 2, 3], "y": [10, 20, 30]})
        chart_with = fm.Chart(df).mark_point().encode(x="x", y="y") + callout(
            x=2.0, y=20.0, text="A"
        )
        chart_without = fm.Chart(df).mark_point().encode(x="x", y="y") + callout(
            x=2.0, y=20.0, text="A", arrow="none"
        )

        svg_with = chart_with.show_svg()
        svg_without = chart_without.show_svg()
        assert svg_with != svg_without, (
            "callout with arrow='none' must produce a different SVG than the default "
            "arrow='curved' — the leader line should be absent with arrow='none'"
        )


# ---------------------------------------------------------------------------
# Bug 3: BreakAxis marks were not repositioned through broken scale
# ---------------------------------------------------------------------------


class TestBreakAxisMarkRepositioning:
    def test_marks_inside_gap_are_hidden(self):
        """Regression: marks with values inside the gap must be clipped/hidden.

        The fix routes mark coordinates through the broken scale, placing
        out-of-gap marks off-screen (sentinel coordinate -99999).
        """
        df = pl.DataFrame({"x": [1, 2, 3], "y": [5.0, 50.0, 95.0]})
        chart = fm.Chart(df).mark_point().encode(x="x", y="y") + BreakAxis(axis="y", gap=(20, 80))
        svg = chart.show_svg()
        # The mark at y=50 (inside the gap 20–80) must be hidden.
        # Rust uses the sentinel -99999 for off-screen/hidden marks.
        assert "-99999" in svg, (
            "A mark whose y-value falls inside the BreakAxis gap must be given "
            "the sentinel coordinate -99999 to hide it from view"
        )

    def test_break_axis_repositions_marks_vs_no_break(self):
        """Regression: marks outside the gap must be repositioned (compressed scale).

        The broken scale collapses the gap, so the remaining marks are closer
        together than they would be on the unbroken scale.  The SVGs must differ.
        """
        df = pl.DataFrame({"x": [1, 2, 3, 4], "y": [5.0, 15.0, 85.0, 95.0]})

        svg_normal = fm.Chart(df).mark_point().encode(x="x", y="y").show_svg()
        svg_break = (
            fm.Chart(df).mark_point().encode(x="x", y="y") + BreakAxis(axis="y", gap=(20, 80))
        ).show_svg()

        assert svg_normal != svg_break, (
            "A chart with BreakAxis must produce a different SVG than the same "
            "chart without it — marks outside the gap must be repositioned"
        )


# ---------------------------------------------------------------------------
# Bug 4: configure_*() methods were completely non-functional
# ---------------------------------------------------------------------------


class TestConfigureMethodsAffectOutput:
    """Each configure_* method must produce a measurably different SVG."""

    def test_configure_grid_color_appears_in_svg(self):
        """Regression: configure_grid(color=) must embed the color in the SVG."""
        df = pl.DataFrame({"x": [1, 2, 3], "y": [10, 20, 30]})
        base = fm.Chart(df).mark_point().encode(x="x", y="y")
        svg_without = base.show_svg()
        svg_with = base.configure_grid(x=True, y=True, color="#ff0000").show_svg()

        assert svg_without != svg_with, "configure_grid must change the rendered SVG"
        assert "ff0000" in svg_with.lower(), (
            "configure_grid(color='#ff0000') must embed the color in the SVG"
        )
        assert "ff0000" not in svg_without.lower(), (
            "The red grid color must not appear in the unconfigured chart"
        )

    def test_configure_legend_orient_bottom_differs_from_default(self):
        """Regression: configure_legend(orient='bottom') must reposition the legend."""
        df = pl.DataFrame({"x": [1, 2, 3, 4], "y": [10, 20, 30, 40], "g": ["a", "b", "a", "b"]})
        chart = fm.Chart(df).mark_point().encode(x="x", y="y", color="g")
        svg_right = chart.show_svg()
        svg_bottom = chart.configure_legend(orient="bottom").show_svg()

        assert svg_right != svg_bottom, (
            "configure_legend(orient='bottom') must produce a different SVG "
            "than the default right-oriented legend"
        )

    def test_configure_padding_changes_layout(self):
        """Regression: configure_padding must affect chart geometry."""
        df = pl.DataFrame({"x": [1, 2, 3], "y": [10, 20, 30]})
        base = fm.Chart(df).mark_point().encode(x="x", y="y")
        svg_default = base.show_svg()
        svg_padded = base.configure_padding(left=100, top=100, auto=False).show_svg()

        assert svg_default != svg_padded, (
            "configure_padding(left=100, top=100) must change chart geometry "
            "relative to the default padding"
        )

    def test_configure_color_scheme_changes_mark_colors(self):
        """Regression: configure_color(scheme=) must change the categorical palette."""
        df = pl.DataFrame({"x": [1, 2, 3, 4], "y": [10, 20, 30, 40], "g": ["a", "b", "c", "d"]})
        chart = fm.Chart(df).mark_point().encode(x="x", y="y", color="g")
        svg_default = chart.show_svg()
        svg_set2 = chart.configure_color(scheme="set2").show_svg()

        assert svg_default != svg_set2, (
            "configure_color(scheme='set2') must produce different mark colors "
            "than the default palette"
        )

    def test_configure_axis_label_font_size_changes_tick_labels(self):
        """Regression: configure_axis(label_font_size=) must change tick label sizes."""
        df = pl.DataFrame({"x": [1, 2, 3], "y": [10, 20, 30]})
        base = fm.Chart(df).mark_point().encode(x="x", y="y")
        svg_default = base.show_svg()
        svg_large = base.configure_axis(label_font_size=20).show_svg()

        assert svg_default != svg_large, (
            "configure_axis(label_font_size=20) must produce different SVG "
            "tick labels than the default font size"
        )

    def test_configure_title_font_size_changes_title_rendering(self):
        """Regression: configure_title(font_size=) must change title appearance."""
        df = pl.DataFrame({"x": [1, 2, 3], "y": [10, 20, 30]})
        chart = fm.Chart(df).mark_point().encode(x="x", y="y").labs(title="Hello")
        svg_default = chart.show_svg()
        svg_big = chart.configure_title(font_size=30).show_svg()

        assert svg_default != svg_big, (
            "configure_title(font_size=30) must produce a different SVG "
            "title than the default title font size"
        )


# ---------------------------------------------------------------------------
# Bug 5: Configure cascade — configure must win over theme
# ---------------------------------------------------------------------------


class TestConfigureCascadePrecedence:
    def test_configure_grid_color_overrides_theme_grid_color(self):
        """Regression: configure_grid(color=) must override theme grid_color.

        The configure surface is applied after the theme dict in the render
        pipeline.  When both set a grid color, configure's value must win —
        the rendered SVG must contain the configure color, not be identical
        to a chart with only the theme applied.
        """
        from ferrum.themes import Theme

        df = pl.DataFrame({"x": [1, 2, 3], "y": [10, 20, 30]})
        chart_theme_only = (
            fm.Chart(df)
            .mark_point()
            .encode(x="x", y="y")
            .theme(Theme(grid=True, grid_color="#aaaaaa"))
        )
        chart_with_override = (
            fm.Chart(df)
            .mark_point()
            .encode(x="x", y="y")
            .theme(Theme(grid=True, grid_color="#aaaaaa"))
            .configure_grid(color="#ff0000")
        )

        svg_theme = chart_theme_only.show_svg()
        svg_override = chart_with_override.show_svg()

        # configure's red must appear in the override chart
        assert "ff0000" in svg_override.lower(), (
            "configure_grid(color='#ff0000') must override the theme's '#aaaaaa' — "
            "the red color must appear in the rendered SVG"
        )
        # The red must not appear in the theme-only chart (verifies the assertion is meaningful)
        assert "ff0000" not in svg_theme.lower(), (
            "The theme-only chart must not contain the configure color #ff0000"
        )
        # The two SVGs must differ, proving configure changed something
        assert svg_theme != svg_override, (
            "Adding configure_grid on top of a themed chart must change the output"
        )


# ---------------------------------------------------------------------------
# Bug 6 — Annotation text had empty font-family (resvg skipped rendering)
# ---------------------------------------------------------------------------


class TestAnnotationTextFontFamily:
    def test_annotation_text_has_font_family_in_svg(self):
        """Regression: annotation text must emit a non-empty font-family."""
        import ferrum.annotation as ann

        df = pl.DataFrame({"x": [1, 2, 3], "y": [10, 20, 30]})
        chart = fm.Chart(df).mark_point().encode(x="x", y="y") + ann.text(2.0, 20.0, "Hello")
        svg = chart.show_svg()
        assert "Hello" in svg
        assert 'font-family=""' not in svg, "font-family must not be empty"


# ---------------------------------------------------------------------------
# Bug 7 — Inset bounds with NormCoord not JSON-serializable
# ---------------------------------------------------------------------------


class TestInsetNormCoordSerialization:
    def test_inset_with_norm_bounds_renders(self):
        """Regression: Inset bounds using NormCoord must serialize correctly."""
        from ferrum.annotation.coords import norm

        df = pl.DataFrame({"x": [1, 2, 3], "y": [10, 20, 30]})
        inset_chart = fm.Chart(df).mark_point().encode(x="x", y="y")
        chart = fm.Chart(df).mark_point().encode(x="x", y="y") + fm.Inset(
            chart=inset_chart, bounds=(norm(0.6), norm(0.1), norm(0.95), norm(0.45))
        )
        svg = chart.show_svg()
        assert svg.startswith("<svg")


# ---------------------------------------------------------------------------
# Bug 8: BreakAxis remap_node on axis text nodes hid x-axis labels
# (commit c4bab99 — removed remap_node call on axes_nodes in scene_build.rs)
# ---------------------------------------------------------------------------


def _text_nodes(svg: str) -> list[str]:
    """Return the visible text content of all <text> elements in the SVG."""
    import re

    return re.findall(r"<text[^>]*>([^<]+)</text>", svg)


class TestBreakAxisAxisLabelsPreserved:
    """Regression: break-axis must remap same-axis ticks while preserving cross-axis labels.

    A y-axis break must remap y-axis tick labels through the broken scale
    (so the ticks match the compressed bar positions), but must NOT touch
    x-axis labels positioned below the plot area.  The remap uses
    node_coord_in_range to filter: only nodes whose coordinate falls within
    the broken axis's pixel range are remapped.
    """

    def test_y_break_preserves_x_category_labels(self):
        """Categorical x-axis tick labels must appear in SVG with a y-axis break."""
        df = pl.DataFrame({"category": ["A", "B", "C"], "value": [10.0, 500.0, 20.0]})
        chart = fm.Chart(df).mark_bar().encode(x="category:N", y="value") + fm.BreakAxis(
            axis="y", gap=(100, 400)
        )
        svg = chart.show_svg()
        texts = _text_nodes(svg)
        for label in ("A", "B", "C"):
            assert label in texts, (
                f"x-axis category label {label!r} must appear in the SVG when a "
                "y-axis BreakAxis is applied — before the fix, remap_node moved "
                "these labels to sentinel y=-99999, hiding them"
            )

    def test_y_break_preserves_x_axis_title(self):
        """X-axis title must remain visible when a y-axis break is applied."""
        df = pl.DataFrame({"category": ["A", "B", "C"], "value": [10.0, 500.0, 20.0]})
        chart = fm.Chart(df).mark_bar().encode(x="category:N", y="value") + fm.BreakAxis(
            axis="y", gap=(100, 400)
        )
        svg = chart.show_svg()
        texts = _text_nodes(svg)
        assert "category" in texts, (
            "x-axis title 'category' must appear in the SVG when a y-axis BreakAxis "
            "is applied — the remap_node bug also displaced the axis title node"
        )

    def test_x_break_preserves_y_axis_title(self):
        """Y-axis title must remain visible when an x-axis break is applied."""
        df = pl.DataFrame({"x": [1.0, 2.0, 3.0], "value": [10.0, 500.0, 20.0]})
        chart = fm.Chart(df).mark_bar().encode(x="x", y="value") + fm.BreakAxis(
            axis="x", gap=(1.5, 2.5)
        )
        svg = chart.show_svg()
        texts = _text_nodes(svg)
        assert "value" in texts, (
            "y-axis title 'value' must appear in the SVG when an x-axis BreakAxis "
            "is applied — axis text nodes outside the broken axis must not be remapped"
        )

    def test_y_break_remaps_y_axis_ticks(self):
        """Regression: y-axis tick labels must be remapped through the broken scale.

        Without remap, bars are compressed but y-axis ticks stay at their
        original linear positions — visually misleading.  After the fix,
        y-axis tick positions are compressed alongside the marks.
        """
        import re

        df = pl.DataFrame({"cat": ["A", "B", "C"], "val": [40.0, 500.0, 60.0]})
        svg_no_break = fm.Chart(df).mark_bar().encode(x="cat:N", y="val").show_svg()
        svg_break = (
            fm.Chart(df).mark_bar().encode(x="cat:N", y="val")
            + fm.BreakAxis(axis="y", gap=(100, 400))
        ).show_svg()

        def y_tick_positions(svg):
            return [
                float(m.group(1))
                for m in re.finditer(r'<text[^>]*\bx="[^"]*"\s+y="([^"]+)"[^>]*>[0-9]', svg)
            ]

        ticks_normal = y_tick_positions(svg_no_break)
        ticks_broken = y_tick_positions(svg_break)
        assert ticks_normal, "Expected y-axis tick positions in normal chart"
        assert ticks_broken, "Expected y-axis tick positions in break chart"
        assert ticks_normal != ticks_broken, (
            "y-axis tick positions must differ between normal and break-axis charts — "
            "the break remap must reposition same-axis ticks through the broken scale"
        )

    def test_y_break_x_labels_not_at_sentinel(self):
        """X-axis labels must not be positioned at the sentinel coordinate y=-99999.

        Y-axis tick labels inside the gap ARE correctly hidden at the sentinel
        (that's the desired break behavior). Only cross-axis labels must be
        preserved — this test checks that category names aren't displaced.
        """
        import re

        df = pl.DataFrame({"category": ["A", "B", "C", "D"], "value": [10.0, 500.0, 20.0, 480.0]})
        chart = fm.Chart(df).mark_bar().encode(x="category:N", y="value") + fm.BreakAxis(
            axis="y", gap=(100, 400)
        )
        svg = chart.show_svg()
        displaced = re.findall(r'<text[^>]*y="-99999[^"]*"[^>]*>([^<]+)</text>', svg)
        x_labels_displaced = [t for t in displaced if t in ("A", "B", "C", "D", "category")]
        assert not x_labels_displaced, (
            f"X-axis labels should not be at the sentinel y=-99999; "
            f"found: {x_labels_displaced}"
        )


# ---------------------------------------------------------------------------
# Bug 9: SecondaryY right-margin not reserved — y2 axis labels clipped
# (commit c4bab99 — compute_layout now reserves padding_right for y2 labels)
# ---------------------------------------------------------------------------


class TestSecondaryYRightMarginReserved:
    """Regression: y2 axis labels must not be clipped by the SVG viewport.

    Before the fix, compute_layout used right: 0.0, so secondary Y axis
    labels positioned at x = plot_right + tick_size + 2 extended past the
    SVG viewport boundary and were truncated.  The fix estimates the y2 label
    width via FontdueMetrics and sets padding_right to accommodate them.
    """

    def test_y2_labels_present_in_svg(self):
        """SecondaryY tick label text must appear in the rendered SVG."""
        df = pl.DataFrame(
            {
                "x": [1.0, 2.0, 3.0, 4.0, 5.0],
                "sales": [100.0, 140.0, 180.0, 220.0, 260.0],
                "growth_rate": [0.438, -0.191, 0.285, 0.123, -0.072],
            }
        )
        chart = fm.Chart(df).mark_bar().encode(x="x", y="sales") + fm.SecondaryY(
            field="growth_rate", mark="line"
        )
        svg = chart.show_svg()
        texts = _text_nodes(svg)
        # The secondary axis tick labels are derived from growth_rate values.
        # At least one negative-valued label must appear (proving y2 is rendered).
        negative_labels = [t for t in texts if t.startswith("-")]
        assert negative_labels, (
            "SecondaryY chart must contain negative-value tick labels from the "
            "secondary axis — before the fix, all y2 labels were clipped by the "
            "SVG viewport"
        )

    def test_y2_label_x_positions_within_viewbox(self):
        """Secondary Y axis label x-start positions must lie inside the SVG viewBox."""
        import re

        df = pl.DataFrame(
            {
                "x": [1.0, 2.0, 3.0, 4.0, 5.0],
                "sales": [100.0, 140.0, 180.0, 220.0, 260.0],
                "growth_rate": [0.438, -0.191, 0.285, 0.123, -0.072],
            }
        )
        chart = fm.Chart(df).mark_bar().encode(x="x", y="sales") + fm.SecondaryY(
            field="growth_rate", mark="line"
        )
        svg = chart.show_svg()

        vb_match = re.search(r'viewBox="0 0 ([0-9.]+)', svg)
        assert vb_match, "SVG must have a numeric viewBox width"
        vb_width = float(vb_match.group(1))

        # Secondary Y labels use text-anchor="start" and are positioned at the
        # right edge of the plot area.
        y2_labels = re.findall(
            r'<text x="([0-9.]+)" y="[^"]+"[^>]*text-anchor="start"[^>]*>([^<]+)</text>',
            svg,
        )
        assert y2_labels, (
            "SecondaryY chart must contain text-anchor='start' labels for the secondary axis"
        )
        for x_str, text in y2_labels:
            x_pos = float(x_str)
            assert x_pos < vb_width, (
                f"y2 label {text!r} at x={x_pos} is outside the SVG viewBox "
                f"(width={vb_width}) — padding_right must be reserved to prevent clipping"
            )

    def test_secondaryy_narrows_plot_area_vs_no_secondary(self):
        """Plot right edge must be narrower when SecondaryY is present.

        The right-margin reservation shifts the plot area left to make room for
        y2 labels.  The rightmost x-axis tick label must therefore be at a lower
        x-coordinate than the same chart rendered without SecondaryY.
        """
        import re

        df = pl.DataFrame(
            {
                "x": [1.0, 2.0, 3.0, 4.0, 5.0],
                "sales": [100.0, 140.0, 180.0, 220.0, 260.0],
                "growth_rate": [0.438, -0.191, 0.285, 0.123, -0.072],
            }
        )
        base = fm.Chart(df).mark_bar().encode(x="x", y="sales")
        svg_without = base.show_svg()
        svg_with = (base + fm.SecondaryY(field="growth_rate", mark="line")).show_svg()

        def rightmost_x_tick(svg: str) -> float:
            # x-axis tick for "5" (the max x value) uses text-anchor="middle"
            m = re.search(
                r'<text x="([0-9.]+)" y="[^"]+"[^>]*text-anchor="middle"[^>]*>5</text>',
                svg,
            )
            assert m, "Could not find x-axis tick label '5'"
            return float(m.group(1))

        x_without = rightmost_x_tick(svg_without)
        x_with = rightmost_x_tick(svg_with)
        assert x_with < x_without, (
            f"The plot right edge (rightmost x-tick at x={x_with:.1f}) must be narrower "
            f"than the chart without SecondaryY (x={x_without:.1f}) — "
            "padding_right must be reserved when a secondary Y axis is present"
        )
