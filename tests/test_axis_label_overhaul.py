"""Integration tests for the axis label layout overhaul.

Covers:
- Collision cascade (wrap → shrink → rotate → cull → elide) doesn't elide
  readable category names on a reasonably-sized chart.
- Faceted charts suppress duplicate y-axis titles (exactly one rendered).
- ``Theme(cull_threshold=...)`` is accepted and the chart renders.
- Dynamic bottom-margin expansion keeps rotated labels inside the SVG viewport.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET

import polars as pl

import ferrum as fm

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
