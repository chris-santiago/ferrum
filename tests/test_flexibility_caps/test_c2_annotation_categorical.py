"""Tests for C2 — annotations anchored to categorical/ordinal axes and
``fm.px``/``fm.norm`` coords accepted by the ``fm.annotate_*`` wrappers.

## What is broken today (TDD RED)

1. ``annotate_vline("cat_a")`` raises ``ValueError`` because ``_coerce_temporal``
   unconditionally calls ``temporal_coord_to_epoch_ms``, which rejects non-ISO-8601
   strings.

2. ``annotate_text(x=fm.px(40), y=fm.norm(0.5), text="hi")`` raises ``TypeError``
   because ``_AnnotationCoord`` excludes ``PixelCoord``/``NormCoord``, so
   ``_coerce_temporal(PixelCoord(40))`` tries ``float(PixelCoord(40))`` which fails.

3. The annotation primitive is serialized with a ``{"category": ...}`` dict that
   must be resolved to a ``{"norm": ...}`` pixel fraction before reaching the Rust
   renderer.

## Tests

- Categorical vline does not raise and annotation node appears in SVG.
- px/norm coordinates accepted by annotate_text, annotate_arrow, annotate_hline,
  annotate_vline.
- Temporal ISO-8601 string annotations still resolve correctly (regression guard).
- Non-ISO string on a temporal axis is treated as a category literal, not mis-parsed.
"""

from __future__ import annotations

import re
import datetime as dt

import polars as pl
import pytest

import ferrum as fm
from ferrum.annotations import annotate_hline, annotate_vline, annotate_text, annotate_arrow


# ---------------------------------------------------------------------------
# SVG helpers
# ---------------------------------------------------------------------------


def _text_contents(svg: str) -> list[str]:
    """Return all text node contents from the SVG."""
    return re.findall(r"<text[^>]*>([^<]+)</text>", svg)


def _has_any_rule(svg: str) -> bool:
    """Return True if the SVG contains a <line> element (mark_rule output)."""
    return bool(re.search(r"<line\b", svg))


def _has_any_text(svg: str) -> bool:
    """Return True if the SVG contains at least one <text> element."""
    return bool(re.search(r"<text\b", svg))


def _annotation_text_count(svg: str) -> int:
    """Return the number of <text> elements in the SVG."""
    return len(re.findall(r"<text\b", svg))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def cat_df() -> pl.DataFrame:
    """Simple DataFrame with a categorical x axis."""
    return pl.DataFrame(
        {
            "cat": ["cat_a", "cat_b", "cat_c"],
            "val": [1.0, 2.0, 3.0],
        }
    )


@pytest.fixture()
def temporal_df() -> pl.DataFrame:
    """DataFrame with a temporal x axis."""
    return pl.DataFrame(
        {
            "date": pl.date_range(dt.date(2020, 1, 1), dt.date(2020, 1, 3), "1d", eager=True),
            "val": [1.0, 2.0, 3.0],
        }
    )


# ---------------------------------------------------------------------------
# C2-A — categorical string coords must not raise
# ---------------------------------------------------------------------------


def test_annotate_vline_categorical_string_does_not_raise(cat_df: pl.DataFrame) -> None:
    """``annotate_vline("cat_a")`` must not raise.

    Today this raises ``ValueError`` from ``temporal_coord_to_epoch_ms`` because
    "cat_a" is not a valid ISO-8601 string.  After the fix, plain strings that are
    not ISO-8601 are treated as ordinal category labels.
    """
    # Must not raise
    ann = annotate_vline("cat_a")
    assert ann is not None, "annotate_vline('cat_a') returned None"


def test_annotate_text_categorical_string_does_not_raise(cat_df: pl.DataFrame) -> None:
    """``annotate_text("cat_a", 1.0, "label")`` must not raise."""
    ann = annotate_text("cat_a", 1.0, "label")
    assert ann is not None


def test_annotate_hline_non_iso_string_does_not_raise() -> None:
    """``annotate_hline("Q2")`` must not raise even though 'Q2' is not ISO-8601."""
    ann = annotate_hline("Q2")
    assert ann is not None


def test_annotate_arrow_categorical_string_does_not_raise() -> None:
    """``annotate_arrow`` with categorical string coords must not raise."""
    ann = annotate_arrow("cat_a", 1.0, "cat_b", 2.0)
    assert ann is not None


def test_annotate_vline_categorical_renders_in_svg(cat_df: pl.DataFrame) -> None:
    """A categorical vline annotation on a nominal x-axis must appear in the SVG.

    The chart has a nominal x-axis (categories: cat_a, cat_b, cat_c).  After the
    annotation is layered, the SVG must contain a <line> element at the cat_b
    band center (approximately the middle third of the plot width).
    """
    base = fm.Chart(cat_df).mark_bar().encode(x="cat:N", y="val:Q")
    ann = annotate_vline("cat_b", stroke="#ff0000")
    composed = base + ann
    svg = composed.show_svg()
    assert _has_any_rule(svg), (
        "SVG must contain at least one <line> element after layering a "
        "categorical annotate_vline.  Check that the annotation node is "
        "emitted and that the category coord is resolved to a valid position."
    )


def test_annotate_text_categorical_renders_in_svg(cat_df: pl.DataFrame) -> None:
    """An annotate_text at a categorical position must produce a text node."""
    base = fm.Chart(cat_df).mark_bar().encode(x="cat:N", y="val:Q")
    ann = annotate_text("cat_a", 1.5, "peak")
    composed = base + ann
    svg = composed.show_svg()
    # The word "peak" must appear in the SVG text nodes.
    texts = _text_contents(svg)
    assert "peak" in texts, f"Expected annotation text 'peak' in SVG text nodes, got: {texts}"


def test_annotate_vline_categorical_position_is_not_center(cat_df: pl.DataFrame) -> None:
    """The categorical vline annotation must be placed at the band, not always at 50%.

    We render two variants: one annotating 'cat_a' (first band, ~16.7% of width)
    and one annotating 'cat_c' (last band, ~83.3% of width).  The x-positions of
    the annotation <line> elements must differ.

    Identification: the annotation uses a distinctive red stroke color (#ff0000)
    not used by any other SVG element.
    """
    ANNOTATION_STROKE = "#ff0000"

    def get_annotation_x1(cat: str) -> str | None:
        base = fm.Chart(cat_df).mark_bar().encode(x="cat:N", y="val:Q")
        ann = annotate_vline(cat, stroke=ANNOTATION_STROKE)
        composed = base + ann
        svg = composed.show_svg()
        # Extract x1 from <line> elements with the annotation stroke color only.
        matches = re.findall(
            rf'<line[^>]+stroke="{re.escape(ANNOTATION_STROKE)}"[^>]*x1="([^"]+)"'
            r'|<line[^>]+x1="([^"]+)"[^>]*stroke="' + re.escape(ANNOTATION_STROKE) + '"',
            svg,
        )
        # re.findall with alternation groups returns tuples; merge non-empty groups
        results = [m[0] or m[1] for m in matches if m[0] or m[1]]
        return results[0] if results else None

    x_a = get_annotation_x1("cat_a")
    x_c = get_annotation_x1("cat_c")

    assert x_a is not None, "No annotation <line> found for cat_a"
    assert x_c is not None, "No annotation <line> found for cat_c"

    # The numeric x positions must differ — different bands
    assert x_a != x_c, (
        f"cat_a and cat_c vline annotations have the same x1 position "
        f"({x_a}). Both category labels must map to their respective band "
        "centers, not a single shared fallback position."
    )

    # cat_a must be to the left of cat_c (ordinal order: a < b < c)
    assert float(x_a) < float(x_c), (
        f"cat_a annotation (x1={x_a}) must be to the left of cat_c (x1={x_c}) "
        "since the ordinal domain is ordered [cat_a, cat_b, cat_c]."
    )


# ---------------------------------------------------------------------------
# C2-B — fm.px / fm.norm accepted by annotate_* wrappers
# ---------------------------------------------------------------------------


def test_annotate_text_px_norm_does_not_raise() -> None:
    """``annotate_text(x=fm.px(40), y=fm.norm(0.5), text="hi")`` must not raise.

    Today this raises ``TypeError`` because ``_AnnotationCoord`` excludes
    ``PixelCoord``/``NormCoord`` and ``_coerce_temporal(PixelCoord(...))`` tries
    ``float(PixelCoord(...))`` which fails.
    """
    ann = annotate_text(x=fm.px(40), y=fm.norm(0.5), text="hi")
    assert ann is not None


def test_annotate_hline_norm_does_not_raise() -> None:
    """``annotate_hline(y=fm.norm(0.5))`` must not raise."""
    ann = annotate_hline(y=fm.norm(0.5))
    assert ann is not None


def test_annotate_vline_px_does_not_raise() -> None:
    """``annotate_vline(x=fm.px(100))`` must not raise."""
    ann = annotate_vline(x=fm.px(100))
    assert ann is not None


def test_annotate_arrow_px_norm_does_not_raise() -> None:
    """``annotate_arrow`` with px/norm coords must not raise."""
    ann = annotate_arrow(
        x1=fm.px(10),
        y1=fm.norm(0.1),
        x2=fm.px(90),
        y2=fm.norm(0.9),
    )
    assert ann is not None


def test_annotate_text_px_norm_renders_in_svg(cat_df: pl.DataFrame) -> None:
    """Pixel/norm coordinates render without error and produce a text node."""
    base = fm.Chart(cat_df).mark_bar().encode(x="cat:N", y="val:Q")
    ann = annotate_text(x=fm.px(40), y=fm.norm(0.5), text="px-norm label")
    composed = base + ann
    svg = composed.show_svg()
    texts = _text_contents(svg)
    assert "px-norm label" in texts, (
        f"Expected 'px-norm label' in SVG text nodes after px/norm annotation, got: {texts}"
    )


def test_annotate_vline_px_renders_in_svg(cat_df: pl.DataFrame) -> None:
    """``annotate_vline(x=fm.px(50))`` renders a <line> element."""
    base = fm.Chart(cat_df).mark_bar().encode(x="cat:N", y="val:Q")
    ann = annotate_vline(x=fm.px(50), stroke="#0f0")
    composed = base + ann
    svg = composed.show_svg()
    assert _has_any_rule(svg), "No <line> element found in SVG for annotate_vline with px() coord."


# ---------------------------------------------------------------------------
# C2-C — temporal ISO-8601 regression guard
# ---------------------------------------------------------------------------


def test_annotate_vline_iso_date_string_still_works(temporal_df: pl.DataFrame) -> None:
    """ISO-8601 date strings are still parsed as temporal, not treated as categories.

    ``annotate_vline("2020-01-02")`` on a temporal axis must still convert to
    epoch-ms and render at the correct date position.
    """
    base = fm.Chart(temporal_df).mark_line().encode(x="date:T", y="val:Q")
    ann = annotate_vline("2020-01-02", stroke="#00f")
    composed = base + ann
    # Must not raise
    svg = composed.show_svg()
    assert _has_any_rule(svg), (
        "ISO-8601 date string annotation must render a <line> element.  "
        "Regression: was working before C2 changes."
    )


def test_annotate_text_iso_date_string_still_works(temporal_df: pl.DataFrame) -> None:
    """ISO-8601 datetime strings still work after C2 changes."""
    base = fm.Chart(temporal_df).mark_line().encode(x="date:T", y="val:Q")
    ann = annotate_text("2020-01-02", 2.0, "midpoint")
    composed = base + ann
    svg = composed.show_svg()
    texts = _text_contents(svg)
    assert "midpoint" in texts, (
        f"ISO date string annotation text 'midpoint' not in SVG, got: {texts}"
    )


def test_annotate_hline_numeric_still_works(temporal_df: pl.DataFrame) -> None:
    """Numeric coordinates still work (regression guard)."""
    base = fm.Chart(temporal_df).mark_line().encode(x="date:T", y="val:Q")
    ann = annotate_hline(y=2.0, stroke="red")
    composed = base + ann
    svg = composed.show_svg()
    assert _has_any_rule(svg), "Numeric annotate_hline must still render after C2 changes."


# ---------------------------------------------------------------------------
# C2-D — non-ISO string on temporal axis is treated as category, not mis-parsed
# ---------------------------------------------------------------------------


def test_non_iso_string_on_temporal_axis_does_not_raise(temporal_df: pl.DataFrame) -> None:
    """A non-ISO string coord like 'peak_date' does not raise even on a temporal chart.

    Contract: the string is treated as a category literal, not silently mis-parsed
    as a date.  The annotation renders without error, falls back to plot center,
    and emits a UserWarning naming the unresolved category and the axis searched.
    """
    base = fm.Chart(temporal_df).mark_line().encode(x="date:T", y="val:Q")
    ann = annotate_vline("peak_date")
    composed = base + ann
    with pytest.warns(UserWarning, match="peak_date"):
        svg = composed.show_svg()
    assert isinstance(svg, str), "show_svg must return a string"


# ---------------------------------------------------------------------------
# C2-E — OrdinalCategoryCoord is exposed from annotation/coords
# ---------------------------------------------------------------------------


def test_ordinal_category_coord_importable() -> None:
    """``OrdinalCategoryCoord`` must be importable from ``ferrum.annotation.coords``."""
    from ferrum.annotation.coords import OrdinalCategoryCoord  # noqa: F401

    coord = OrdinalCategoryCoord(value="cat_a")
    assert coord.value == "cat_a"


def test_ordinal_category_coord_is_frozen() -> None:
    """``OrdinalCategoryCoord`` must be immutable (frozen dataclass)."""
    from ferrum.annotation.coords import OrdinalCategoryCoord

    coord = OrdinalCategoryCoord(value="x")
    with pytest.raises((TypeError, AttributeError)):
        coord.value = "y"  # type: ignore[misc]


def test_iso_string_does_not_become_ordinal_category_coord() -> None:
    """An ISO-8601 string like '2020-01-01' must NOT be coerced to OrdinalCategoryCoord.

    It should remain as an epoch-ms float in the annotation primitive.
    """
    from ferrum.annotations import annotate_vline
    from ferrum.annotation.coords import OrdinalCategoryCoord

    ann = annotate_vline("2020-01-01")
    # The underlying primitive x1 should be a float (epoch-ms), not OrdinalCategoryCoord
    prim = ann._annotation_primitive
    if prim is not None:
        assert not isinstance(prim.x1, OrdinalCategoryCoord), (
            "ISO-8601 string '2020-01-01' was incorrectly coerced to "
            "OrdinalCategoryCoord; it should be epoch-ms float."
        )


def test_non_iso_string_becomes_ordinal_category_coord() -> None:
    """A non-ISO string like 'cat_a' must become an OrdinalCategoryCoord in the primitive."""
    from ferrum.annotations import annotate_vline
    from ferrum.annotation.coords import OrdinalCategoryCoord

    ann = annotate_vline("cat_a")
    prim = ann._annotation_primitive
    if prim is not None:
        assert isinstance(prim.x1, OrdinalCategoryCoord), (
            f"Non-ISO string 'cat_a' should be coerced to OrdinalCategoryCoord "
            f"in the primitive, got {type(prim.x1)}"
        )
        assert prim.x1.value == "cat_a"


# ---------------------------------------------------------------------------
# C2-F — annotate_arrow coord-coercion paths (regression for _coerce_coord)
#
# These tests specifically exercise the lines in annotate_arrow that call
# _coerce_coord(x1/y1/x2/y2) and _coerce_coord_to_numeric(…).  Before the C2
# fix these lines called the now-removed _coerce_temporal, causing a
# NameError on this path.  The tests also cover the labelled-arrow branch
# (annotate_text delegation) with categorical and ISO-string coordinates.
# ---------------------------------------------------------------------------


def test_annotate_arrow_iso_string_coords_do_not_raise() -> None:
    """``annotate_arrow`` with ISO-8601 date strings must not raise.

    Exercises _coerce_coord(x1), _coerce_coord(y1), _coerce_coord(x2),
    _coerce_coord(y2) with the ISO-8601 branch (string → epoch-ms float).
    This is the arrow-specific regression path for the removed _coerce_temporal.
    """
    ann = annotate_arrow("2020-01-01", 1.0, "2020-06-01", 3.0)
    assert ann is not None


def test_annotate_arrow_iso_string_coords_store_epoch_ms_in_primitive() -> None:
    """ISO-8601 strings in annotate_arrow must resolve to epoch-ms floats in the primitive.

    Covers the _coerce_coord ISO-string branch inside annotate_arrow.
    """
    from ferrum.annotation.coords import OrdinalCategoryCoord

    ann = annotate_arrow("2020-01-01", 1.0, "2020-06-01", 3.0)
    prim = ann._annotation_primitive
    if prim is not None:
        assert not isinstance(prim.x1, OrdinalCategoryCoord), (
            "ISO-8601 string '2020-01-01' in annotate_arrow must store a float "
            "(epoch-ms), not OrdinalCategoryCoord."
        )
        assert isinstance(prim.x1, float), (
            f"Expected float for ISO-8601 x1 in arrow primitive, got {type(prim.x1)}"
        )


def test_annotate_arrow_non_iso_string_coords_store_ordinal_in_primitive() -> None:
    """Non-ISO strings in annotate_arrow must resolve to OrdinalCategoryCoord in the primitive.

    Covers the _coerce_coord non-ISO branch inside annotate_arrow.
    """
    from ferrum.annotation.coords import OrdinalCategoryCoord

    ann = annotate_arrow("cat_start", 0.0, "cat_end", 1.0)
    prim = ann._annotation_primitive
    if prim is not None:
        assert isinstance(prim.x1, OrdinalCategoryCoord), (
            f"Non-ISO string 'cat_start' in annotate_arrow must produce "
            f"OrdinalCategoryCoord in the primitive, got {type(prim.x1)}"
        )
        assert prim.x1.value == "cat_start"
        assert isinstance(prim.x2, OrdinalCategoryCoord), (
            f"Non-ISO string 'cat_end' in annotate_arrow must produce "
            f"OrdinalCategoryCoord for x2, got {type(prim.x2)}"
        )
        assert prim.x2.value == "cat_end"


def test_annotate_arrow_with_label_and_categorical_coords_does_not_raise() -> None:
    """``annotate_arrow`` with a label and categorical coords must not raise.

    The labelled branch delegates to annotate_text(lx, ly, …) with an
    OrdinalCategoryCoord — this exercises the full arrow + text composition
    path through _coerce_coord.
    """
    ann = annotate_arrow("cat_a", 1.0, "cat_b", 2.0, label="transition", label_side="end")
    assert ann is not None


def test_annotate_arrow_with_label_and_iso_string_coords_does_not_raise() -> None:
    """Labelled annotate_arrow with ISO-8601 start/end coords must not raise.

    Exercises the labelled-arrow path with ISO-8601 string coords, confirming
    that annotate_text receives a float (epoch-ms) coordinate correctly.
    """
    ann = annotate_arrow("2020-01-01", 1.0, "2020-06-01", 3.0, label="H1 2020", label_side="start")
    assert ann is not None
