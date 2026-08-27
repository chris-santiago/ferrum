"""Regression tests for finding R2: y-axis ``label_angle`` was silently inert.

Per-channel ``fm.Axis(label_angle=...)`` and chart-level
``configure_axis(label_angle=...)`` rotated x tick labels but did nothing on
y — the Rust layout/render path only handled the x arm of the axis. This
file proves the fix on the y axis: rotation renders, ``label_angle=0``
stays byte-identical to omitting it, secondary-y honors the override, and
rotated y labels reserve a narrower band than long unrotated ones (the
transpose semantics the fix restores). An x-side sanity check guards that
the existing x behavior was not disturbed.

``test_per_channel_y_label_angle_rotates`` was used to prove the bug RED
before the Rust fix via the stash-rebuild protocol (``git stash push --
crates/`` -> rebuild -> run -> ``git stash pop`` -> rebuild); see the task
report for the recorded RED/GREEN runs.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET

import polars as pl
import pytest

import ferrum as fm
from ferrum.structural import SecondaryY

_SVG_NS = "{http://www.w3.org/2000/svg}"


def _svg_root(svg: str) -> ET.Element:
    return ET.fromstring(svg)


def _lines(root: ET.Element) -> list[ET.Element]:
    return root.findall(".//" + _SVG_NS + "line")


def _texts(root: ET.Element) -> list[ET.Element]:
    return root.findall(".//" + _SVG_NS + "text")


def _rotated_texts(root: ET.Element, degrees: float) -> list[ET.Element]:
    needle = f"rotate({degrees:g}"
    return [t for t in _texts(root) if needle in (t.get("transform") or "")]


def _y_domain_line(root: ET.Element) -> ET.Element | None:
    """The primary (left) y-axis domain line: a tall, near-vertical line.

    Mirrors ``_x_domain_line`` in ``test_phase_12_axis_legend.py`` but for
    the y axis: a vertical line (``x1 == x2``) spanning most of the plot
    height. Tick marks are short vertical segments protruding from it, so
    candidates are filtered to long spans; when a secondary y axis is also
    present its domain line sits further right, so the *leftmost* candidate
    is the primary y axis.
    """
    candidates: list[tuple[float, ET.Element]] = []
    for ln in _lines(root):
        try:
            x1 = float(ln.get("x1", "nan"))
            x2 = float(ln.get("x2", "nan"))
            y1 = float(ln.get("y1", "nan"))
            y2 = float(ln.get("y2", "nan"))
        except ValueError:
            continue
        if abs(x1 - x2) <= 2.0 and (y2 - y1) > 100.0:
            candidates.append((x1, ln))
    if not candidates:
        return None
    return min(candidates, key=lambda c: c[0])[1]


@pytest.fixture()
def scatter_df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            "y": list(range(10)),
        }
    )


class TestPerChannelYLabelAngle:
    def test_per_channel_y_label_angle_rotates(self, scatter_df: pl.DataFrame) -> None:
        """fm.Y(axis=fm.Axis(label_angle=-45)) changes the SVG and rotates y ticks.

        x is left untouched, so any ``rotate(-45`` text node in the rotated
        render is unambiguously a y-axis tick label. This is the RED-proof
        case: pre-fix, the y arm was inert and this render was byte-identical
        to the unrotated baseline.
        """
        unrotated = fm.Chart(scatter_df).mark_point().encode(x="x", y="y").to_svg()
        rotated = (
            fm.Chart(scatter_df)
            .mark_point()
            .encode(x="x", y=fm.Y("y", axis=fm.Axis(label_angle=-45)))
            .to_svg()
        )
        assert rotated != unrotated, "y label_angle=-45 must change the rendered SVG"
        y_rotated = _rotated_texts(_svg_root(rotated), -45)
        assert y_rotated, "expected rotate(-45 ...) on y tick labels"
        assert not _rotated_texts(_svg_root(unrotated), -45), (
            "unrotated baseline must have no rotate(-45 ...) text nodes"
        )

    def test_y_label_angle_zero_is_byte_identical_to_omitted(
        self, scatter_df: pl.DataFrame
    ) -> None:
        """Explicit label_angle=0 on y must render identically to omitting axis config."""
        omitted = fm.Chart(scatter_df).mark_point().encode(x="x", y="y").to_svg()
        explicit_zero = (
            fm.Chart(scatter_df)
            .mark_point()
            .encode(x="x", y=fm.Y("y", axis=fm.Axis(label_angle=0.0)))
            .to_svg()
        )
        assert explicit_zero == omitted, (
            "label_angle=0 on y must be byte-identical to omitting label_angle"
        )

    def test_x_side_sanity_still_rotates(self, scatter_df: pl.DataFrame) -> None:
        """Existing x label_angle behavior is unaffected by the y-axis fix."""
        svg = (
            fm.Chart(scatter_df)
            .mark_point()
            .encode(x=fm.X("x", axis=fm.Axis(label_angle=-45)), y="y")
            .to_svg()
        )
        assert _rotated_texts(_svg_root(svg), -45), (
            "per-channel label_angle on x must still rotate x tick labels by -45"
        )


class TestSecondaryYLabelAngle:
    def test_secondary_y_axis_honors_label_angle(self) -> None:
        """SecondaryY(axis=fm.Axis(label_angle=-45)) rotates the secondary axis's labels."""
        df = pl.DataFrame(
            {
                "month": ["Jan", "Feb", "Mar", "Apr", "May", "Jun"],
                "revenue": [125000, 138500, 112000, 161000, 183000, 172000],
                "growth_rate": [0.0, 0.107, -0.191, 0.438, 0.137, -0.066],
            }
        )
        base = fm.Chart(df).mark_bar().encode(x="month:N", y="revenue:Q")

        flat = (base + SecondaryY(field="growth_rate", mark="line")).to_svg()
        rotated = (
            base
            + SecondaryY(
                field="growth_rate",
                mark="line",
                axis=fm.Axis(label_angle=-45),
            )
        ).to_svg()

        assert rotated != flat, "secondary-y label_angle=-45 must change the rendered SVG"
        assert _rotated_texts(_svg_root(rotated), -45), (
            "expected rotate(-45 ...) on the secondary y axis's tick labels"
        )
        assert not _rotated_texts(_svg_root(flat), -45), (
            "secondary-y baseline (no label_angle) must have no rotate(-45 ...) text nodes"
        )


class TestYLabelAngleBandWidth:
    def test_rotated_y_labels_reserve_narrower_band(self) -> None:
        """Rotated -90 y tick labels widen the plot area vs long unrotated ones.

        Long horizontal y tick labels reserve width equal to their text
        width; rotating them -90 (angle-aware, transposed extent) reserves
        width equal to their text height instead, which is much smaller for
        long category names. A narrower reserved band pushes the left edge
        of the plot area (the primary y-axis domain line) further left,
        i.e. the plot area gets wider.
        """
        df = pl.DataFrame(
            {
                "category": [
                    "extremely_long_category_label_number_one",
                    "extremely_long_category_label_number_two",
                    "extremely_long_category_label_number_three",
                ],
                "value": [10, 20, 30],
            }
        )
        flat = (
            fm.Chart(df)
            .mark_bar()
            .encode(x="value:Q", y="category:N")
            .properties(width=500, height=300)
            .to_svg()
        )
        rotated = (
            fm.Chart(df)
            .mark_bar()
            .encode(
                x="value:Q",
                y=fm.Y("category:N", axis=fm.Axis(label_angle=-90)),
            )
            .properties(width=500, height=300)
            .to_svg()
        )
        assert rotated != flat

        flat_domain = _y_domain_line(_svg_root(flat))
        rotated_domain = _y_domain_line(_svg_root(rotated))
        assert flat_domain is not None, "expected a primary y-axis domain line (flat)"
        assert rotated_domain is not None, "expected a primary y-axis domain line (rotated)"

        flat_left_edge = float(flat_domain.get("x1"))
        rotated_left_edge = float(rotated_domain.get("x1"))
        assert rotated_left_edge < flat_left_edge - 10.0, (
            f"rotated (-90) y labels should reserve a narrower band, pushing the plot "
            f"area's left edge further left: flat={flat_left_edge}, rotated={rotated_left_edge}"
        )
