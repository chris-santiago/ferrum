"""Integration tests for the axis label layout overhaul.

Covers:
- Collision cascade (wrap → shrink → rotate → cull → elide) doesn't elide
  readable category names on a reasonably-sized chart.
- Faceted charts suppress duplicate y-axis titles (exactly one rendered).
- ``Theme(cull_threshold=...)`` is accepted and the chart renders.
- Dynamic bottom-margin expansion keeps rotated labels inside the SVG viewport.
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET

import polars as pl

import ferrum as fm
from tests._snapshots import decode_png_rgba, rasterize_svg

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _all_text_content(root: ET.Element) -> list[str]:
    """Return every text node and tspan text visible in *root*."""
    texts: list[str] = []
    for elem in root.iter():
        if elem.text and elem.text.strip():
            texts.append(elem.text.strip())
        if elem.tail and elem.tail.strip():
            texts.append(elem.tail.strip())
    return texts


def _svg_root(svg: str) -> ET.Element:
    return ET.fromstring(svg)


# ---------------------------------------------------------------------------
# Test 1: Nine snake_case categories — no elision
# ---------------------------------------------------------------------------

_NINE_CATEGORIES = [
    "trivial_baseline",
    "negative_prompt",
    "persona_constrained",
    "minimal_context",
    "none",
    "generic_coder",
    "real_agent_config",
    "python_coder",
    "long_directive",
]


def test_nine_snake_case_categories_no_elision() -> None:
    """600×400 bar chart with 9 snake_case categories → no ellipsis elision.

    The collision cascade (wrap → shrink → rotate → cull) should handle the
    overlap before falling back to elision. All 9 category names must appear
    in the SVG, either as single text elements or split across multi-line tspan
    children.
    """
    df = pl.DataFrame(
        {
            "preamble": _NINE_CATEGORIES,
            "value": list(range(1, 10)),
        }
    )

    chart = fm.Chart(df).mark_bar().encode(x="preamble:N", y="value:Q")
    svg = chart.properties(width=600, height=400).to_svg()

    assert "<svg" in svg, "Expected a valid SVG document"

    root = _svg_root(svg)
    all_text = _all_text_content(root)
    joined = " ".join(all_text)

    # No Unicode ellipsis (…) should appear in any text element
    assert "…" not in joined, (
        f"Ellipsis (…) found in SVG text — elision occurred. Text elements: {all_text!r}"
    )

    # Every category name must be present (possibly split across tspan lines).
    # A wrapped label like "trivial_\nbaseline" appears as two separate text
    # nodes, so we check that each word-fragment from the category is present
    # rather than requiring the full underscore-joined string.
    for category in _NINE_CATEGORIES:
        # The label may be wrapped on underscore boundaries; check that the
        # first "word" of each label appears.
        first_word = category.split("_")[0]
        assert any(first_word in t for t in all_text), (
            f"Category {category!r} (first word {first_word!r}) not found "
            f"in SVG text elements: {all_text!r}"
        )


# ---------------------------------------------------------------------------
# Test 2: Faceted chart has exactly one y-axis title
# ---------------------------------------------------------------------------


def test_faceted_chart_single_y_title() -> None:
    """Faceted chart with 2 columns → only one y-axis title in the SVG."""
    df = pl.DataFrame(
        {
            "x": ["a", "b", "c"] * 4,
            "y": [1.0, 2.0, 3.0, 1.5, 2.5, 3.5, 0.5, 2.2, 3.1, 1.8, 2.8, 3.8],
            "group": ["G1", "G1", "G1", "G1", "G1", "G1", "G2", "G2", "G2", "G2", "G2", "G2"],
        }
    )

    svg = (
        fm.Chart(df)
        .mark_bar()
        .encode(x="x:N", y="y:Q")
        .facet("group", ncols=2)
        .labs(y="My Y Title")
        .to_svg()
    )

    assert "<svg" in svg, "Expected a valid SVG document"

    root = _svg_root(svg)
    all_text = _all_text_content(root)
    occurrences = sum(1 for t in all_text if "My Y Title" in t)

    assert occurrences == 1, (
        f"Expected exactly 1 occurrence of 'My Y Title' in SVG text elements; "
        f"found {occurrences}. All text: {all_text!r}"
    )


# ---------------------------------------------------------------------------
# Test 3: cull_threshold accepted by Theme
# ---------------------------------------------------------------------------


def test_cull_threshold_theme_parameter() -> None:
    """Theme(cull_threshold=5) doesn't raise and the chart renders."""
    theme = fm.Theme(cull_threshold=5)

    df = pl.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
    svg = fm.Chart(df).mark_point().encode(x="x:Q", y="y:Q").theme(theme).to_svg()

    assert "<svg" in svg, "Expected a valid SVG document"
    assert len(svg) > 200, "SVG suspiciously small — chart may not have rendered"


# ---------------------------------------------------------------------------
# Test 4: Rotated labels not clipped (dynamic margin)
# ---------------------------------------------------------------------------

_LONG_LABELS = [f"this_is_a_very_long_label_{i}" for i in range(1, 7)]


def test_rotated_labels_not_clipped() -> None:
    """Chart with 6 long labels: no text element should extend below viewBox height."""
    df = pl.DataFrame(
        {
            "category": _LONG_LABELS,
            "value": [10, 20, 15, 25, 30, 18],
        }
    )

    svg = (
        fm.Chart(df)
        .mark_bar()
        .encode(x="category:N", y="value:Q")
        .properties(width=400, height=300)
        .to_svg()
    )

    assert "<svg" in svg, "Expected a valid SVG document"

    root = _svg_root(svg)

    # Extract viewBox height (fallback to height attribute)
    viewbox = root.get("viewBox") or ""
    if viewbox:
        # viewBox="0 0 W H"
        parts = viewbox.split()
        viewport_height = float(parts[3]) if len(parts) == 4 else 300.0
    else:
        viewport_height = float(root.get("height", "300").rstrip("px"))

    # Find all text elements and check their y coordinates
    text_elements = root.findall(".//{http://www.w3.org/2000/svg}text")
    for text_el in text_elements:
        y_attr = text_el.get("y")
        if y_attr is None:
            continue
        try:
            y_val = float(y_attr)
        except ValueError:
            continue
        # Allow a small tolerance (a few pixels for descenders / tspan offsets)
        assert y_val <= viewport_height + 20, (
            f"Text element y={y_val} exceeds viewport height {viewport_height} "
            f"(text={text_el.text!r}). Labels may be clipped."
        )


# ---------------------------------------------------------------------------
# Regression: rotated x-axis tick-label anchoring + x-title clearance
# (branch fix/rotated-axis-label-anchor, 2026-06).
#
# Bug 1: rotated bottom x-tick labels were emitted with text-anchor="middle",
#        so they pivoted about their horizontal centre and the right half of
#        each label swung up into the plot panel.
# Bug 2: the x-axis title was placed using a FLAT label height, so for rotated
#        labels (which extend much further below the axis) the title overlapped
#        the labels.
# Fix:   rotated bottom labels are end-anchored (pinned at the trailing edge)
#        with an angle-aware vertical gap; the x-title drops below the true
#        rotated-label extent (its placement is now angle-aware).
# ---------------------------------------------------------------------------

# Medium-length categories that rotate cleanly at -45 (no elision) AND are long
# enough that rotation expands the bottom margin, shifting the x-title downward.
_ROT_CATEGORIES = [
    "configuration_one",
    "configuration_two",
    "configuration_three",
    "configuration_four",
    "configuration_five",
    "configuration_six",
]

_SVG_NS = "{http://www.w3.org/2000/svg}"


def _text_nodes(svg: str) -> list[tuple[str, str, str | None, float | None]]:
    """Return ``(content, transform, text-anchor, y)`` for every ``<text>``."""
    nodes: list[tuple[str, str, str | None, float | None]] = []
    for t in _svg_root(svg).findall(".//" + _SVG_NS + "text"):
        y = t.get("y")
        nodes.append(
            (
                (t.text or "").strip(),
                t.get("transform") or "",
                t.get("text-anchor"),
                float(y) if y is not None else None,
            )
        )
    return nodes


def _rotated_bar_svg(
    label_angle: int | None,
    *,
    categories: list[str] = _ROT_CATEGORIES,
    width: int = 600,
    height: int = 320,
) -> str:
    df = pl.DataFrame(
        {"category": categories, "value": [10, 20, 15, 25, 30, 18][: len(categories)]}
    )
    chart = (
        fm.Chart(df)
        .mark_bar()
        .encode(x="category:N", y="value:Q")
        .properties(width=width, height=height)
    )
    if label_angle is not None:
        chart = chart.configure_axis(label_angle=label_angle)
    return chart.to_svg()


def test_rotated_x_labels_are_end_anchored() -> None:
    """Regression (Bug 1): rotated bottom x-tick labels must be end-anchored.

    They were ``text-anchor="middle"``, which pivoted them about their centre and
    pushed the right half into the plot. Rotated labels must pin at the trailing
    (right) edge — ``text-anchor="end"``.
    """
    svg = _rotated_bar_svg(label_angle=-45)
    rotated = [
        (content, anchor)
        for (content, transform, anchor, _y) in _text_nodes(svg)
        if "rotate(" in transform and content in _ROT_CATEGORIES
    ]
    assert rotated, "expected rotated x-tick labels at label_angle=-45"
    assert all(anchor == "end" for _, anchor in rotated), (
        f"rotated x-tick labels must be end-anchored, got {rotated}"
    )


def test_flat_x_labels_stay_middle_anchored() -> None:
    """Sibling guard: unrotated x-tick labels must stay centre-anchored.

    The end-anchor change applies only when the label is rotated; the common
    (flat) case must remain ``text-anchor="middle"``.
    """
    df = pl.DataFrame({"g": ["a", "b", "c"], "v": [1, 2, 3]})
    svg = fm.Chart(df).mark_bar().encode(x="g:N", y="v:Q").to_svg()
    flat = [
        (content, anchor)
        for (content, transform, anchor, _y) in _text_nodes(svg)
        if "rotate(" not in transform and content in {"a", "b", "c"}
    ]
    assert flat, "expected flat x-tick labels"
    assert all(anchor == "middle" for _, anchor in flat), (
        f"flat x-tick labels must stay middle-anchored, got {flat}"
    )


def test_rotated_x_labels_do_not_disturb_y_axis_title() -> None:
    """Guard: the anchor fix touches only x-tick labels, not the y-axis title.

    The y-axis title is rotated -90 and must remain centre-anchored.
    """
    svg = _rotated_bar_svg(label_angle=-45)
    y_titles = [
        (content, anchor)
        for (content, transform, anchor, _y) in _text_nodes(svg)
        if "rotate(-90" in transform
    ]
    assert y_titles, "expected a rotated (-90) y-axis title"
    assert all(anchor == "middle" for _, anchor in y_titles), (
        f"y-axis title must stay middle-anchored, got {y_titles}"
    )


def _has_blank_row_between(svg: str, y_a: float, y_b: float) -> bool:
    """True if *svg*'s own rasterized canvas has at least one scanline, in
    the row range spanning baselines *y_a* and *y_b* (either order), with no
    rendered ink in any column.

    A lower text line's descenders and an upper line's ascent both sit
    strictly inside the span between their own baselines, so a genuine
    vertical gap between two stacked lines always leaves at least one fully
    blank row inside that span; two lines whose ink touches or overlaps
    leave none. This reads the real rasterized glyph ink -- the same
    `resvg_py` raster `test_padding_auto_default.py` uses for its overhang
    margins -- rather than an estimated font metric, so it also accounts
    for an end-anchored rotated label's ink dropping well below its pivot.
    """
    rows, _width, height = decode_png_rgba(rasterize_svg(svg))
    background = rows[0][0]
    lo = max(0, math.floor(min(y_a, y_b)))
    hi = min(height - 1, math.ceil(max(y_a, y_b)))
    for y in range(lo, hi + 1):
        if not any(px != background and px[3] != 0 for px in rows[y]):
            return True
    return False


# Longer than `_ROT_CATEGORIES`: four underscore-joined words each, so the
# flat rendering wraps onto several lines (not just two) and the -45 rotated
# rendering's end-anchored diagonal sweep drops well past the flat band --
# deep enough that a title placed at the flat band's height would ride into
# it. `_ROT_CATEGORIES` is too short for that: its rotated ink ends above
# where its own flat labels wrap to, so a title using the flat position
# never reaches it either.
_TITLE_CLEARANCE_CATEGORIES = [
    "configuration_one_alpha_bravo",
    "configuration_two_alpha_bravo",
    "configuration_three_alpha_bravo",
    "configuration_four_alpha_bravo",
    "configuration_five_alpha_bravo",
    "configuration_six_alpha_bravo",
]
_TITLE_CLEARANCE_WORDS = {
    word for category in _TITLE_CLEARANCE_CATEGORIES for word in category.split("_")
} | set(_TITLE_CLEARANCE_CATEGORIES)


def test_x_axis_title_clears_rotated_labels() -> None:
    """Regression (Bug 2): the x-axis title must clear the rotated labels.

    The title was positioned from a flat label height, so its y was
    independent of the label angle and it overlapped rotated labels. Three
    invariants now hold:

    1. The title sits below every rotated label's pivot.
    2. The rotated title's y does not sit above the flat title's y. The two
       are placed by the same panel-bottom-plus-title-metrics formula, so in
       a correct render they land at the identical y; a title derived from a
       flat (single-line) label height instead of the true rotated extent
       would place the rotated title higher, at a smaller y than the flat
       title.
    3. The title's rendered ink never touches the last drawn x-tick-label
       line's rendered ink -- for the flat rendering (whose multi-word
       category names wrap onto several lines) and for the rotated one
       (whose end-anchored -45 labels drop well below their shared pivot)
       alike.
    """
    flat_svg = _rotated_bar_svg(
        label_angle=None, categories=_TITLE_CLEARANCE_CATEGORIES, width=800, height=400
    )
    rot_svg = _rotated_bar_svg(
        label_angle=-45, categories=_TITLE_CLEARANCE_CATEGORIES, width=800, height=400
    )
    flat_nodes = _text_nodes(flat_svg)
    rot_nodes = _text_nodes(rot_svg)

    def _x_title_y(nodes: list[tuple[str, str, str | None, float | None]]) -> float:
        return next(
            y
            for (content, transform, _a, y) in nodes
            if content == "category" and "rotate(" not in transform and y is not None
        )

    def _last_tick_label_line_y(
        nodes: list[tuple[str, str, str | None, float | None]],
    ) -> float:
        return max(
            y
            for (content, _transform, _a, y) in nodes
            if content in _TITLE_CLEARANCE_WORDS and y is not None
        )

    flat_title_y = _x_title_y(flat_nodes)
    rot_title_y = _x_title_y(rot_nodes)
    rot_pivots = [
        y
        for (content, transform, _a, y) in rot_nodes
        if "rotate(" in transform and content in _TITLE_CLEARANCE_CATEGORIES and y is not None
    ]

    assert rot_pivots, "expected rotated x-tick labels at label_angle=-45"
    # (1) Title below every rotated label pivot.
    assert rot_title_y > max(rot_pivots), (
        f"x-title.y={rot_title_y} must sit below rotated label pivots (max pivot={max(rot_pivots)})"
    )
    # (2) The rotated title must not sit above the flat title (measured
    # margin: exactly zero -- both are placed by the same formula).
    assert rot_title_y >= flat_title_y, (
        f"rotated x-title.y={rot_title_y} must not sit above the flat "
        f"x-title.y={flat_title_y}; a title derived from a flat label "
        f"height would place the rotated title higher"
    )
    # (3) The title's rendered ink must not touch the last drawn tick-label
    # line's rendered ink, flat and rotated alike. Comparing baselines (or a
    # rotated line's shared pivot) cannot show this: two baselines can sit
    # far enough apart on paper and still have their painted glyphs touch
    # once each line's own ascent/descent -- or, for the rotated line, the
    # real diagonal drop below its pivot -- is accounted for. This checks
    # the rasterized rows for a real gap instead.
    flat_last_line_y = _last_tick_label_line_y(flat_nodes)
    rot_last_line_y = _last_tick_label_line_y(rot_nodes)
    assert _has_blank_row_between(flat_svg, flat_last_line_y, flat_title_y), (
        f"flat x-title ink (y={flat_title_y}) must not touch the last wrapped "
        f"label line's ink (y={flat_last_line_y})"
    )
    assert _has_blank_row_between(rot_svg, rot_last_line_y, rot_title_y), (
        f"rotated x-title ink (y={rot_title_y}) must not touch the last rotated "
        f"label line's ink, including its drop below the shared pivot (y={rot_last_line_y})"
    )
