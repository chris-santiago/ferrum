"""Guard test for issue #25: square glyphs in faceted charts stay within panel clips.

Investigation summary (2026-06-20): issue #25 claimed that square glyphs land
outside the panel clip in faceted charts. The investigation found this does NOT
reproduce. A square's half-extent is r*0.8 (≈2.708 px at the default size),
which is SMALLER than a circle's radius (≈3.385 px at the same size). Squares
therefore protrude less beyond the panel edge than circles do, and in practice
every data square's bounding box is fully contained within its panel clip rect.

This test locks in that correct behavior. It will FAIL if a future change causes
any faceted square glyph bbox to exceed its owning panel clip rect, making it a
real discriminator for a regression. Issue #25 is closed as not-a-bug with this
test as evidence.

Shape palette order: index 0 = Circle (<circle>), index 1 = Square (<rect> with
equal w and h). Using ≥2 shape categories ensures the second category (palette
index 1) maps to Square.
"""

from __future__ import annotations

import re
from typing import NamedTuple

import polars as pl
import pytest

import ferrum as fm


# ---------------------------------------------------------------------------
# SVG parsing helpers
# ---------------------------------------------------------------------------


class ClipRect(NamedTuple):
    clip_id: str
    x: float
    y: float
    w: float
    h: float

    @property
    def x1(self) -> float:
        return self.x

    @property
    def y1(self) -> float:
        return self.y

    @property
    def x2(self) -> float:
        return self.x + self.w

    @property
    def y2(self) -> float:
        return self.y + self.h

    def contains_point(self, cx: float, cy: float) -> bool:
        """True if the point (cx, cy) is inside this clip rect (inclusive)."""
        return self.x1 <= cx <= self.x2 and self.y1 <= cy <= self.y2

    def contains_bbox(self, rx: float, ry: float, rw: float, rh: float) -> bool:
        """True if the rect (rx, ry, rw, rh) is fully within this clip rect."""
        return rx >= self.x1 and ry >= self.y1 and rx + rw <= self.x2 and ry + rh <= self.y2


class GlyphRect(NamedTuple):
    x: float
    y: float
    w: float
    h: float

    @property
    def cx(self) -> float:
        return self.x + self.w / 2

    @property
    def cy(self) -> float:
        return self.y + self.h / 2

    @property
    def half_extent(self) -> float:
        """Half the side length (assumes square glyph w == h)."""
        return self.w / 2


def _parse_clip_rects(svg: str) -> list[ClipRect]:
    """Extract all <clipPath><rect .../></clipPath> entries from an SVG string."""
    clips = []
    for m in re.finditer(
        r'<clipPath id="([^"]+)"><rect x="([0-9.]+)" y="([0-9.]+)" '
        r'width="([0-9.]+)" height="([0-9.]+)"',
        svg,
    ):
        clip_id, x, y, w, h = m.groups()
        clips.append(ClipRect(clip_id, float(x), float(y), float(w), float(h)))
    return clips


def _parse_data_square_glyphs(svg: str, clips: list[ClipRect]) -> list[GlyphRect]:
    """Extract data <rect> glyphs whose center falls inside a panel clip.

    Filtering strategy:
    - Skip background/panel background rects (large width, typically > 20 px).
    - Skip legend swatches: they are small rects but their CENTER falls outside
      every panel clip rect (legend sits in the right margin / bottom).
    - The remaining rects whose center is inside a panel clip are data glyphs.

    The legend swatch is also larger than data glyphs (w≈8 vs w≈5.4), but we
    use the center-inside-clip criterion as the primary filter so the test
    remains robust to theme size changes.
    """
    glyphs = []
    for m in re.finditer(
        r'<rect x="([0-9.]+)" y="([0-9.]+)" width="([0-9.]+)" height="([0-9.]+)"',
        svg,
    ):
        rx, ry, rw, rh = (float(v) for v in m.groups())
        # Skip large structural rects (panel backgrounds, full SVG background).
        if rw > 20 or rh > 20:
            continue
        glyph = GlyphRect(rx, ry, rw, rh)
        # Only include glyphs whose center is inside some panel clip.
        if any(c.contains_point(glyph.cx, glyph.cy) for c in clips):
            glyphs.append(glyph)
    return glyphs


def _parse_data_circle_radius(svg: str, clips: list[ClipRect]) -> float | None:
    """Return the radius of a data circle whose center is inside a panel clip.

    Returns the first match (all data circles share the same radius for the same
    size encoding).
    """
    for m in re.finditer(
        r'<circle cx="([0-9.]+)" cy="([0-9.]+)" r="([0-9.]+)"',
        svg,
    ):
        cx, cy, r = float(m.group(1)), float(m.group(2)), float(m.group(3))
        if any(c.contains_point(cx, cy) for c in clips):
            return r
    return None


def _owning_clip(glyph: GlyphRect, clips: list[ClipRect]) -> ClipRect | None:
    """Return the clip whose panel contains the glyph's center, or None."""
    for c in clips:
        if c.contains_point(glyph.cx, glyph.cy):
            return c
    return None


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def faceted_shape_df() -> pl.DataFrame:
    """Faceted scatter with 2 shape categories at axis extremes.

    Two panels (A and B), each with:
    - grp='a' (Circle, palette index 0) at x in {0.0, 1.0}, y in {0.0, 1.0}
    - grp='b' (Square, palette index 1) at x in {0.0, 1.0}, y in {0.0, 1.0}

    Data is placed at axis extremes so glyphs land near panel edges, providing
    the strongest possible test of clipping behavior.
    """
    return pl.DataFrame(
        {
            "x": [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
            "y": [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0],
            "grp": ["a", "a", "b", "b", "a", "a", "b", "b"],
            "panel": ["A", "A", "A", "A", "B", "B", "B", "B"],
        }
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSquareGlyphClipInFacets:
    """Guard tests for issue #25: square glyphs stay inside panel clip rects.

    Uses EXPLICIT ncols=2 so this test is invariant to any KG-6 default-layout
    change (which only affects col-only facets without an explicit ncols).
    """

    def _render_svg(self, df: pl.DataFrame) -> tuple[str, list[ClipRect]]:
        svg = (
            fm.Chart(df)
            .mark_point()
            .encode(x="x:Q", y="y:Q", shape="grp:N")
            .facet(col="panel", ncols=2)  # EXPLICIT ncols — invariant to KG-6 default change
            .to_svg()
        )
        clips = _parse_clip_rects(svg)
        return svg, clips

    def test_panel_clips_are_present(self, faceted_shape_df: pl.DataFrame) -> None:
        """Two panel clips must be emitted for a two-panel faceted chart."""
        _, clips = self._render_svg(faceted_shape_df)
        assert len(clips) == 2, (
            f"Expected 2 panel <clipPath> rects for 2 facet panels; got {len(clips)}: {clips}"
        )

    def test_data_square_glyphs_are_found(self, faceted_shape_df: pl.DataFrame) -> None:
        """Data square (<rect>) glyphs must be detected in the SVG (sanity check).

        'b' is shape category 2 (palette index 1 = Square), with 2 data points
        per panel, so at least 4 data square glyphs must be present.
        """
        svg, clips = self._render_svg(faceted_shape_df)
        squares = _parse_data_square_glyphs(svg, clips)
        assert len(squares) >= 4, (
            f"Expected ≥4 data square glyphs (grp='b' in 2 panels × 2 rows); "
            f"got {len(squares)}: {squares}. "
            "If 0, the shape encoding may not be mapping 'b' to Square, or the "
            "filtering logic excludes data glyphs incorrectly."
        )

    def test_every_square_bbox_inside_panel_clip(self, faceted_shape_df: pl.DataFrame) -> None:
        """Core guard: every data square's bounding box is fully contained in its panel clip.

        This assertion is a real discriminator: if a square's bbox ever extends
        beyond its clip rect boundary (e.g. due to a coordinate-offset bug),
        the test fails. Data points are placed at axis extremes (x=0, x=1, y=0,
        y=1) so glyphs land at the panel edges where clipping issues would appear.
        """
        svg, clips = self._render_svg(faceted_shape_df)
        squares = _parse_data_square_glyphs(svg, clips)
        assert squares, "No data square glyphs found — cannot validate clipping."

        violations: list[str] = []
        for glyph in squares:
            clip = _owning_clip(glyph, clips)
            assert clip is not None, (
                f"Square glyph {glyph} has center ({glyph.cx:.3f}, {glyph.cy:.3f}) "
                "outside all panel clips — unexpected filter miss."
            )
            if not clip.contains_bbox(glyph.x, glyph.y, glyph.w, glyph.h):
                violations.append(
                    f"Square bbox [{glyph.x:.3f}, {glyph.y:.3f}, "
                    f"{glyph.x + glyph.w:.3f}, {glyph.y + glyph.h:.3f}] "
                    f"exceeds panel clip [{clip.x1:.3f}, {clip.y1:.3f}, "
                    f"{clip.x2:.3f}, {clip.y2:.3f}] "
                    f"(clip_id={clip.clip_id!r})"
                )

        assert not violations, (
            "Issue #25 regression: square glyph(s) exceed their panel clip rect.\n"
            + "\n".join(violations)
        )

    def test_square_half_extent_less_than_circle_radius(
        self, faceted_shape_df: pl.DataFrame
    ) -> None:
        """A square's half-extent must be ≤ a same-size circle's radius.

        This is the geometric evidence that squares are LESS prone to panel-edge
        overflow than circles. The brief (§investigation) notes half-extent = r*0.8;
        this test documents and locks in that relationship.
        """
        svg, clips = self._render_svg(faceted_shape_df)
        squares = _parse_data_square_glyphs(svg, clips)
        circle_r = _parse_data_circle_radius(svg, clips)

        assert squares, "No data square glyphs found."
        assert circle_r is not None, (
            "No data circle found — cannot compare square half-extent to circle radius. "
            "Ensure the shape fixture has grp='a' (Circle) data points."
        )

        for glyph in squares:
            assert glyph.half_extent <= circle_r + 1e-9, (
                f"Square half-extent {glyph.half_extent:.4f} exceeds circle radius "
                f"{circle_r:.4f}. Expected half_extent ≤ radius (spec: half_extent = r*0.8). "
                "A regression in shape sizing would cause squares to protrude beyond "
                "the panel clip even when circles stay inside."
            )

    def test_discriminator_would_fail_on_oversized_square(self) -> None:
        """Verify test_every_square_bbox_inside_panel_clip is a real discriminator.

        Constructs a synthetic scenario with a square that overflows its clip and
        confirms the validation logic raises AssertionError. This proves the guard
        assertion is not trivially true (i.e. it CAN fail).
        """
        # Build a fake clip that is too small for the square glyph.
        # Square at (100, 100) with w=h=10 → bbox ends at (110, 110).
        # Clip ends at (108, 108) → right/bottom sides overflow by 2px each.
        glyph = GlyphRect(x=100.0, y=100.0, w=10.0, h=10.0)
        tight_clip = ClipRect(
            clip_id="ferrum-clip-tight",
            x=98.0,
            y=98.0,
            w=10.0,  # x2 = 108 < 110 (glyph right edge)
            h=10.0,  # y2 = 108 < 110 (glyph bottom edge)
        )

        # The glyph center (105, 105) IS inside the clip (98..108), so it would
        # be classified as a data glyph. But the bbox overflows.
        assert tight_clip.contains_point(glyph.cx, glyph.cy), (
            "Test setup error: glyph center should be inside the tight clip."
        )
        assert not tight_clip.contains_bbox(glyph.x, glyph.y, glyph.w, glyph.h), (
            "Test setup error: glyph bbox should NOT fit in the tight clip."
        )

        # Run the containment check and confirm it flags a violation.
        violations: list[str] = []
        clip = _owning_clip(glyph, [tight_clip])
        if clip is not None and not clip.contains_bbox(glyph.x, glyph.y, glyph.w, glyph.h):
            violations.append("overflow detected")

        assert violations, (
            "Discriminator check: expected a violation for an oversized square, "
            "but none was detected. The guard assertion may be trivially true."
        )
