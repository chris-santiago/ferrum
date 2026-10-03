"""Golden SVG tests for the declarative configuration surface.

Covers:
- Annotation primitives: text, line, rect, arrow, span
- Structural features: SecondaryY, BreakAxis, Inset
- Configure methods: configure_axis, configure_grid
- A combined chart exercising configure + annotations + structural

Run with ``FERRUM_UPDATE_GOLDENS=1`` to regenerate the on-disk goldens;
the default behavior is byte-identical comparison.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

import ferrum as fm
from ferrum.annotation.container import Annotate
from ferrum.annotation.coords import norm, px
from ferrum.annotation.primitives import arrow, line, rect, span, text
from ferrum.structural import BreakAxis, Inset, SecondaryY
from tests._snapshots import check_or_update_svg_golden


GOLDENS_DIR = Path(__file__).parent / "goldens" / "configure"


def _check_or_update(name: str, svg: str) -> None:
    check_or_update_svg_golden(GOLDENS_DIR, name, svg)


# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------


@pytest.fixture()
def sample_df():
    return pl.DataFrame(
        {
            "x": [1.0, 2.0, 3.0, 4.0, 5.0],
            "y": [10.0, 25.0, 15.0, 30.0, 20.0],
            "revenue": [100.0, 200.0, 150.0, 300.0, 250.0],
            "category": ["A", "B", "A", "B", "A"],
        }
    )


# ---------------------------------------------------------------------------
# Annotation primitive goldens
# ---------------------------------------------------------------------------


class TestAnnotationGoldens:
    def test_text_annotation(self, sample_df):
        chart = fm.Chart(sample_df).mark_point().encode(x="x", y="y") + text(
            3.0, 25.0, "Peak", font_size=14
        )
        _check_or_update("annotation_text.svg", chart.to_svg())

    def test_line_annotation(self, sample_df):
        chart = fm.Chart(sample_df).mark_point().encode(x="x", y="y") + line(
            1.0, 20.0, 5.0, 20.0, stroke="#e45756", stroke_width=2
        )
        _check_or_update("annotation_line.svg", chart.to_svg())

    def test_rect_annotation(self, sample_df):
        chart = fm.Chart(sample_df).mark_point().encode(x="x", y="y") + rect(
            2.0, 10.0, 4.0, 30.0, fill="#4c78a8", opacity=0.2
        )
        _check_or_update("annotation_rect.svg", chart.to_svg())

    def test_arrow_annotation(self, sample_df):
        chart = fm.Chart(sample_df).mark_point().encode(x="x", y="y") + arrow(
            2.0, 28.0, 3.0, 25.0, stroke="#333"
        )
        _check_or_update("annotation_arrow.svg", chart.to_svg())

    def test_span_annotation(self, sample_df):
        chart = fm.Chart(sample_df).mark_point().encode(x="x", y="y") + span(
            "x", 2.0, 4.0, fill="#f58518", opacity=0.15
        )
        _check_or_update("annotation_span.svg", chart.to_svg())

    def test_multiple_annotations(self, sample_df):
        annotations = Annotate(
            [
                text(3.0, 30.0, "Annotated"),
                line(1.0, 20.0, 5.0, 20.0, stroke="#999"),
                rect(1.5, 10.0, 2.5, 30.0, fill="#4c78a8", opacity=0.1),
            ]
        )
        chart = fm.Chart(sample_df).mark_point().encode(x="x", y="y") + annotations
        _check_or_update("annotation_multiple.svg", chart.to_svg())

    def test_pixel_coord_annotation(self, sample_df):
        chart = fm.Chart(sample_df).mark_point().encode(x="x", y="y") + text(
            px(50), px(30), "Fixed position"
        )
        _check_or_update("annotation_pixel_coords.svg", chart.to_svg())

    def test_norm_coord_annotation(self, sample_df):
        chart = fm.Chart(sample_df).mark_point().encode(x="x", y="y") + text(
            norm(0.5), norm(0.1), "Center-top"
        )
        _check_or_update("annotation_norm_coords.svg", chart.to_svg())


# ---------------------------------------------------------------------------
# Structural feature goldens
# ---------------------------------------------------------------------------


class TestStructuralGoldens:
    def test_secondary_y(self, sample_df):
        chart = fm.Chart(sample_df).mark_point().encode(x="x", y="y") + SecondaryY(
            "revenue", color="#e45756"
        )
        _check_or_update("structural_secondary_y.svg", chart.to_svg())

    def test_break_axis_slash(self, sample_df):
        chart = fm.Chart(sample_df).mark_point().encode(x="x", y="y") + BreakAxis(
            axis="y", gap=(18, 22), break_style="slash"
        )
        _check_or_update("structural_break_axis_slash.svg", chart.to_svg())

    def test_break_axis_zigzag(self, sample_df):
        chart = fm.Chart(sample_df).mark_point().encode(x="x", y="y") + BreakAxis(
            axis="y", gap=(18, 22), break_style="zigzag"
        )
        _check_or_update("structural_break_axis_zigzag.svg", chart.to_svg())

    def test_break_axis_wave(self, sample_df):
        chart = fm.Chart(sample_df).mark_point().encode(x="x", y="y") + BreakAxis(
            axis="y", gap=(18, 22), break_style="wave"
        )
        _check_or_update("structural_break_axis_wave.svg", chart.to_svg())

    def test_break_axis_bar_clipped_into_segments(self):
        """Extent clipping (spec §4 H4): the crossing bar renders as two rects.

        The guide's worked example — a bar to 900 under a ``(150, 850)`` gap is
        drawn 0–150 and 850–900 — rather than the single full-plot rect that
        stretched through the break before. Point-mark break goldens above stay
        byte-identical; this golden is where the clipped geometry lives.
        """
        df = pl.DataFrame(
            {"category": ["A", "B", "C", "D", "E"], "value": [72.0, 68.0, 75.0, 900.0, 81.0]}
        )
        chart = fm.Chart(df).mark_bar().encode(x="category:N", y="value:Q") + BreakAxis(
            axis="y", gap=(150, 850)
        )
        _check_or_update("structural_break_axis_bar.svg", chart.to_svg())

    def test_break_axis_area_clipped_into_segments(self):
        """An area spanning the break is clipped into one band per segment."""
        df = pl.DataFrame({"x": [1.0, 2.0, 3.0, 4.0], "value": [50.0, 900.0, 950.0, 60.0]})
        chart = fm.Chart(df).mark_area().encode(x="x:Q", y="value:Q") + BreakAxis(
            axis="y", gap=(150, 850)
        )
        _check_or_update("structural_break_axis_area.svg", chart.to_svg())

    def test_break_axis_segment_arrow_clipped_into_segments(self):
        """#231: an arrowed segment crossing a break gap splits into honest pieces.

        Mirrors ``test_break_axis_bar_clipped_into_segments`` above: a
        ``mark_segment(arrow=True)`` shaft crossing a ``(150, 850)`` gap renders
        as two shaft pieces (no stroke spanning the gap), with the arrowhead
        riding the piece that keeps the tip (wave-2 T6, #230-1).
        """
        df = pl.DataFrame({"x1": [1.0], "y1": [20.0], "x2": [3.0], "y2": [900.0]})
        chart = fm.Chart(df).mark_segment(arrow=True, arrow_size=10).encode(
            x="x1:Q", y="y1:Q", x2="x2:Q", y2="y2:Q"
        ) + BreakAxis(axis="y", gap=(150, 850))
        _check_or_update("structural_break_axis_segment_arrow.svg", chart.to_svg())

    def test_inset(self, sample_df):
        inset_chart = fm.Chart(sample_df).mark_point().encode(x="x", y="revenue")
        chart = fm.Chart(sample_df).mark_point().encode(x="x", y="y") + Inset(
            chart=inset_chart, bounds=(0.6, 0.1, 0.95, 0.45)
        )
        _check_or_update("structural_inset.svg", chart.to_svg())


# ---------------------------------------------------------------------------
# Configure method goldens
# ---------------------------------------------------------------------------


class TestConfigureGoldens:
    def test_configure_axis_rotation(self, sample_df):
        chart = (
            fm.Chart(sample_df)
            .mark_point()
            .encode(x="category:N", y="y")
            .configure_axis(label_angle=-45)
        )
        _check_or_update("configure_axis_rotation.svg", chart.to_svg())

    def test_configure_grid(self, sample_df):
        chart = (
            fm.Chart(sample_df)
            .mark_point()
            .encode(x="x", y="y")
            .configure_grid(y=True, color="#ddd")
        )
        _check_or_update("configure_grid.svg", chart.to_svg())

    def test_grid_minor(self, sample_df):
        """Minor gridlines via ferrum.Grid (item 18) on a continuous chart.

        Exercises the full Python->Rust minor path: a theme-level
        ``Grid(minor=True)`` renders minor gridlines (lighter/thinner, here an
        explicit minor color) between the scale-projected major gridlines on
        both continuous axes. Inspect the PNG to confirm minors sit between
        majors and coincide with the projected grid.
        """
        theme = fm.Theme(grid=True).update(
            grid=fm.Grid(major=True, minor=True, minor_color="#e8e8e8")
        )
        chart = fm.Chart(sample_df).mark_point().encode(x="x", y="y").theme(theme)
        _check_or_update("grid_minor.svg", chart.to_svg())


# ---------------------------------------------------------------------------
# Composite mark style goldens
# ---------------------------------------------------------------------------


class TestCompositeMarkStyleGoldens:
    def test_density_styled(self):
        """Overlapping translucent KDE areas via mark_density(opacity=0.4).

        Renders two density curves (one per group) with 40% opacity, so the
        overlap region shows through both fills. The resulting SVG must contain
        rgba() fill values with the 0.4 alpha component. Inspect the PNG to
        confirm two translucent overlapping area fills are visible.
        """
        import polars as pl

        df = pl.DataFrame(
            {
                "value": [
                    1.0,
                    1.5,
                    2.0,
                    2.5,
                    3.0,
                    3.5,
                    4.0,
                    4.5,
                    5.0,
                    5.5,
                    2.5,
                    3.0,
                    3.5,
                    4.0,
                    4.5,
                    5.0,
                    5.5,
                    6.0,
                    6.5,
                    7.0,
                ],
                "group": ["a"] * 10 + ["b"] * 10,
            }
        )
        chart = (
            fm.Chart(df)
            .mark_density(multiple="layer", opacity=0.4)
            .encode(x="value:Q", color="group:N")
            .properties(width=400, height=250)
        )
        svg = chart.to_svg()
        assert "rgba(" in svg, "Expected rgba() fill in SVG for translucent density areas"
        _check_or_update("density_styled.svg", svg)


# ---------------------------------------------------------------------------
# Combined golden
# ---------------------------------------------------------------------------


class TestCombinedGolden:
    def test_full_combined_chart(self, sample_df):
        """Combined chart with configure + annotations + secondary Y."""
        chart = (
            fm.Chart(sample_df)
            .mark_point()
            .encode(x="x", y="y")
            .configure_axis(label_angle=-30)
            .configure_grid(y=True, color="#eee")
            + text(3.0, 30.0, "Peak value", font_size=12)
            + line(1.0, 20.0, 5.0, 20.0, stroke="#999", dash=[4, 2])
            + SecondaryY("revenue", mark="line", color="#e45756")
        )
        _check_or_update("combined_full.svg", chart.to_svg())
