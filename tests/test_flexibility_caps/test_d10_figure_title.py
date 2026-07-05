"""Regression tests for defect D10 — composite figure-level title/subtitle/caption.

Spec (ferrum-spec.md §4/§6): a composite chart (vconcat / hconcat / facet)
must support `composite.properties(title=, subtitle=, caption=)` that renders:
  - the title (and subtitle) ONCE, above the WHOLE composed figure
  - the caption ONCE, below the WHOLE composed figure
  - per-panel child titles independently (not overwritten by the figure title)

Current (broken) behavior
--------------------------
1. ``properties(title=)`` on a composite fans the title out to EVERY child via
   ``_ChartLike.properties`` → ``_rebuild_with_charts(lambda c: c.properties(**kwargs))``.
   Each child renders its own title band, so the title appears N times (once
   per panel) rather than once around the whole figure.

2. ``properties(subtitle=, caption=)`` on a composite raises ``TypeError``
   because ``Chart.properties()`` does not accept ``subtitle`` or ``caption``
   keyword arguments, and the composite fan-out delegates directly to
   ``Chart.properties``.

3. Calling ``composite.properties(title="Figure Title")`` overwrites per-panel
   child titles that were set before the composition — ``Panel A`` title
   disappears even though the intent is only to set a figure-level wrapper.

Root cause
----------
Python layer: ``_ChartLike.properties`` (``src/ferrum/composition.py:245``)
has no concept of "figure-level" vs "panel-level" properties.  It fans
``**kwargs`` indiscriminately to every child chart.

Rust layer: the string-based 1D SVG compositors (vertical / horizontal stacking,
previously in the now-deleted ``render/compositor.rs``) accept only SVG
strings and a spacing value; they have no facility to inject a figure-wide
title band before the panel grid or a caption band below it.

The grid compositor (previously in the now-deleted ``render/grid_compose.rs``)
similarly has no figure-level title or caption parameter.

For a fix to land, both layers must change:
  - Python: ``_ChartLike`` needs a ``figure_title`` / ``figure_subtitle`` /
    ``figure_caption`` store (or an overloaded ``properties`` that distinguishes
    figure-level from per-panel kwargs), and the render call must pass those
    values through.
  - Rust: each string compositor needs an optional title-band (above) and
    caption-band (below) parameter so that a single text node wraps the whole
    output SVG.

These tests assert the INTENDED behavior. All tests are expected to FAIL until
the fix lands (TDD RED).

Test surface
------------
  D10-T1  vconcat .properties(title=, subtitle=, caption=) → each appears once
  D10-T2  hconcat .properties(title=, subtitle=, caption=) → each appears once
  D10-T3  figure title appears once even with N panels (not N copies)
  D10-T4  per-panel child titles survive when a figure-level title is set
  D10-T5  faceted chart .properties(title=, subtitle=, caption=) → each once
  D10-T6  subtitle keyword on composite .properties() does not raise TypeError
  D10-T7  caption keyword on composite .properties() does not raise TypeError
  D10-T8  figure title is positioned above all panels (before any panel title
          in document order)
  D10-T9  caption is positioned below all panels (after all panel content in
          document order)
"""

from __future__ import annotations

import xml.etree.ElementTree as ET

import polars as pl
import pytest

import ferrum as fm

_SVG_NS = "{http://www.w3.org/2000/svg}"


def _root_width(svg: str) -> float:
    """Return the composed SVG's intrinsic width as a float."""
    return float(ET.fromstring(svg).get("width"))


def _find_text_node(svg: str, text: str) -> ET.Element:
    """Return the ``<text>`` element whose text content equals *text*.

    Raises ``AssertionError`` when no such node exists so failing position
    assertions surface a clear message instead of an ``AttributeError``.
    """
    root = ET.fromstring(svg)
    for el in root.iter(f"{_SVG_NS}text"):
        if el.text == text:
            return el
    raise AssertionError(f"no <text> node with content {text!r} found in SVG")


def _figure_title_node(svg: str, text: str) -> ET.Element:
    """Return the figure-level title node (font-size 16, font-weight 600).

    The figure title band is emitted by the Rust compositor with a distinct
    16px / 600-weight style; per-panel child titles use a different style, so
    matching on both attributes isolates the figure title.
    """
    node = _find_text_node(svg, text)
    assert node.get("font-size") == "16", (
        f"expected figure title font-size 16; got {node.get('font-size')!r}"
    )
    assert node.get("font-weight") == "600", (
        f"expected figure title font-weight 600; got {node.get('font-weight')!r}"
    )
    return node


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def two_charts():
    """Two minimal charts suitable for vconcat / hconcat."""
    df = pl.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
    c1 = fm.Chart(df).mark_point().encode(x="x", y="y")
    c2 = fm.Chart(df).mark_bar().encode(x="x", y="y")
    return c1, c2


@pytest.fixture
def two_charts_with_panel_titles():
    """Two charts that each carry their own panel-level title."""
    df = pl.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
    c1 = fm.Chart(df, title="Panel A").mark_point().encode(x="x", y="y")
    c2 = fm.Chart(df, title="Panel B").mark_bar().encode(x="x", y="y")
    return c1, c2


@pytest.fixture
def facet_chart():
    """Minimal faceted chart with two panels."""
    df = pl.DataFrame(
        {
            "x": [1, 2, 3, 1, 2, 3],
            "y": [4, 5, 6, 7, 8, 9],
            "g": ["a", "a", "a", "b", "b", "b"],
        }
    )
    return fm.Chart(df).mark_point().encode(x="x", y="y").facet(col="g")


# ---------------------------------------------------------------------------
# D10-T1: vconcat — figure title, subtitle, and caption each appear exactly once
# ---------------------------------------------------------------------------


def test_d10_t1_vconcat_figure_title_subtitle_caption_appear_once(two_charts):
    """vconcat .properties(title=, subtitle=, caption=) renders each text once.

    EXPECTED TO FAIL (TDD RED):
    - ``subtitle=`` raises TypeError because Chart.properties does not accept it.
    - Even if it did not raise, both title and subtitle would appear twice
      (once per child panel) instead of once.
    - ``caption=`` is not supported at all.
    """
    c1, c2 = two_charts
    composed = (c1 & c2).properties(
        title="Figure Title",
        subtitle="Figure Subtitle",
        caption="Figure Caption",
    )
    svg = composed.to_svg()

    assert svg.count("Figure Title") == 1, (
        f"Expected figure title to appear exactly once; got {svg.count('Figure Title')} copies. "
        "Current behavior: title is fanned to every child panel, appearing N times."
    )
    assert svg.count("Figure Subtitle") == 1, (
        f"Expected figure subtitle to appear exactly once; got {svg.count('Figure Subtitle')} copies."
    )
    assert svg.count("Figure Caption") == 1, (
        f"Expected figure caption to appear exactly once; got {svg.count('Figure Caption')} copies."
    )


# ---------------------------------------------------------------------------
# D10-T2: hconcat — figure title, subtitle, and caption each appear exactly once
# ---------------------------------------------------------------------------


def test_d10_t2_hconcat_figure_title_subtitle_caption_appear_once(two_charts):
    """hconcat .properties(title=, subtitle=, caption=) renders each text once.

    EXPECTED TO FAIL (TDD RED): same fan-out failure as vconcat.
    """
    c1, c2 = two_charts
    composed = (c1 | c2).properties(
        title="Figure Title",
        subtitle="Figure Subtitle",
        caption="Figure Caption",
    )
    svg = composed.to_svg()

    assert svg.count("Figure Title") == 1, (
        f"Expected figure title once in hconcat; got {svg.count('Figure Title')} copies."
    )
    assert svg.count("Figure Subtitle") == 1, (
        f"Expected figure subtitle once in hconcat; got {svg.count('Figure Subtitle')} copies."
    )
    assert svg.count("Figure Caption") == 1, (
        f"Expected figure caption once in hconcat; got {svg.count('Figure Caption')} copies."
    )


# ---------------------------------------------------------------------------
# D10-T3: title appears exactly once regardless of panel count
# ---------------------------------------------------------------------------


def test_d10_t3_title_appears_once_with_three_panels():
    """A three-panel vconcat renders the figure title exactly once.

    EXPECTED TO FAIL (TDD RED): title is fanned to all 3 children, appearing
    3 times in the SVG instead of once around the whole figure.
    """
    df = pl.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
    charts = [fm.Chart(df).mark_point().encode(x="x", y="y") for _ in range(3)]
    from ferrum.composition import VConcatChart

    composed = VConcatChart(charts).properties(title="Three-Panel Title")
    svg = composed.to_svg()

    assert svg.count("Three-Panel Title") == 1, (
        f"Three-panel vconcat: expected title once, got {svg.count('Three-Panel Title')} copies. "
        "Current behavior fans title to every child, producing N copies."
    )


# ---------------------------------------------------------------------------
# D10-T4: per-panel child titles survive when a figure-level title is set
# ---------------------------------------------------------------------------


def test_d10_t4_per_panel_titles_survive_figure_title(two_charts_with_panel_titles):
    """Child panel titles are preserved when composite .properties(title=) is called.

    EXPECTED TO FAIL (TDD RED): the current fan-out calls child.properties(title=)
    on every child, overwriting their own titles with the figure-level title.
    Panel A and Panel B titles both disappear.
    """
    c1, c2 = two_charts_with_panel_titles
    composed = (c1 & c2).properties(title="Figure Title")
    svg = composed.to_svg()

    # Figure-level title wraps the whole figure (once).
    assert svg.count("Figure Title") == 1, (
        f"Expected figure title once; got {svg.count('Figure Title')} copies."
    )
    # Per-panel titles must still appear independently.
    assert "Panel A" in svg, (
        "Panel A title was overwritten by figure-level .properties(title=). "
        "Per-panel titles must survive independently."
    )
    assert "Panel B" in svg, (
        "Panel B title was overwritten by figure-level .properties(title=). "
        "Per-panel titles must survive independently."
    )


# ---------------------------------------------------------------------------
# D10-T5: faceted chart — figure-level title, subtitle, caption each once
# ---------------------------------------------------------------------------


def test_d10_t5_facet_figure_title_subtitle_caption_appear_once(facet_chart):
    """Faceted chart .properties(title=, subtitle=, caption=) renders each text once.

    EXPECTED TO FAIL (TDD RED):
    - Faceted chart is a ``Chart`` object with ``_facet`` set; ``properties``
      exists on ``Chart`` but does not accept ``subtitle`` or ``caption``.
    - A faceted chart renders the title inside the facet grid (once, via the
      single-Chart title path) — which is closer to correct, but subtitle
      and caption are still unsupported.
    """
    composed = facet_chart.properties(
        title="Facet Figure Title",
        subtitle="Facet Subtitle",
        caption="Facet Caption",
    )
    svg = composed.to_svg()

    assert svg.count("Facet Figure Title") == 1, (
        f"Expected facet figure title once; got {svg.count('Facet Figure Title')} copies."
    )
    assert svg.count("Facet Subtitle") == 1, (
        f"Expected facet subtitle once; got {svg.count('Facet Subtitle')} copies."
    )
    assert svg.count("Facet Caption") == 1, (
        f"Expected facet caption once; got {svg.count('Facet Caption')} copies."
    )


# ---------------------------------------------------------------------------
# D10-T6: subtitle keyword on composite .properties() does not raise TypeError
# ---------------------------------------------------------------------------


def test_d10_t6_subtitle_on_composite_properties_does_not_raise(two_charts):
    """composite.properties(subtitle=...) must not raise TypeError.

    EXPECTED TO FAIL (TDD RED): ``_ChartLike.properties`` fans kwargs to
    ``Chart.properties``, which does not accept ``subtitle=``, causing
    ``TypeError: Chart.properties() got an unexpected keyword argument 'subtitle'``.
    """
    c1, c2 = two_charts
    # Must not raise.
    result = (c1 & c2).properties(subtitle="A subtitle")
    svg = result.to_svg()
    assert "A subtitle" in svg, "subtitle text must appear in the rendered SVG"


# ---------------------------------------------------------------------------
# D10-T7: caption keyword on composite .properties() does not raise TypeError
# ---------------------------------------------------------------------------


def test_d10_t7_caption_on_composite_properties_does_not_raise(two_charts):
    """composite.properties(caption=...) must not raise TypeError.

    EXPECTED TO FAIL (TDD RED): ``caption`` is not a parameter on
    ``Chart.properties`` at all — neither the composite fan-out nor the
    per-chart path accepts it, so a ``TypeError`` is raised.
    """
    c1, c2 = two_charts
    # Must not raise.
    result = (c1 & c2).properties(caption="A caption note")
    svg = result.to_svg()
    assert "A caption note" in svg, "caption text must appear in the rendered SVG"


# ---------------------------------------------------------------------------
# D10-T8: figure title appears above all panels (document order)
# ---------------------------------------------------------------------------


def _text_node_y(svg: str, content: str) -> float:
    """Return the ``y`` coordinate of the ``<text>`` node whose text is *content*."""
    import re

    for attrs, text in re.findall(r"<text\s+([^>]*)>([^<]*)</text>", svg):
        if text.strip() == content:
            m = re.search(r'y="([^"]+)"', attrs)
            if m:
                return float(m.group(1))
    raise AssertionError(f"no <text> node with content {content!r}")


def test_d10_t8_figure_title_appears_above_panel_content(two_charts_with_panel_titles):
    """The figure-level title must render above all panel content.

    The composite render path emits the figure chrome into one scene and shifts
    the panels down, so the assertion is on *visual position* (the figure title's
    y is above both per-panel titles) rather than SVG byte order — the string
    compositor happened to prepend the chrome band, but z-order/byte-order is a
    render mechanism, not the figure-level-placement contract this test guards.
    """
    c1, c2 = two_charts_with_panel_titles
    composed = (c1 & c2).properties(title="Figure Title")
    svg = composed.to_svg()

    assert "Figure Title" in svg, "Figure Title must be present"
    assert "Panel A" in svg, "Panel A must be present"
    assert "Panel B" in svg, "Panel B must be present"

    figure_y = _text_node_y(svg, "Figure Title")
    panel_a_y = _text_node_y(svg, "Panel A")
    panel_b_y = _text_node_y(svg, "Panel B")

    assert figure_y < panel_a_y, (
        f"Figure title (y={figure_y}) must render above Panel A title (y={panel_a_y})."
    )
    assert figure_y < panel_b_y, (
        f"Figure title (y={figure_y}) must render above Panel B title (y={panel_b_y})."
    )


# ---------------------------------------------------------------------------
# D10-T9: caption appears after all panel content (document order)
# ---------------------------------------------------------------------------


def test_d10_t9_caption_appears_after_all_panel_content(two_charts_with_panel_titles):
    """The figure-level caption must follow all panel body content in SVG order.

    EXPECTED TO FAIL (TDD RED): ``caption`` is not supported at all today; this
    test will raise TypeError before reaching the SVG assertion.

    When the fix lands, the caption text node must appear at a later byte offset
    than both per-panel title/body regions.
    """
    c1, c2 = two_charts_with_panel_titles
    # Must not raise TypeError.
    composed = (c1 & c2).properties(title="Figure Title", caption="Source: test data")
    svg = composed.to_svg()

    assert "Figure Title" in svg, "Figure Title must be present"
    assert "Source: test data" in svg, "Caption text must be present"
    assert "Panel A" in svg, "Panel A must be present"
    assert "Panel B" in svg, "Panel B must be present"

    caption_pos = svg.index("Source: test data")
    panel_a_pos = svg.index("Panel A")
    panel_b_pos = svg.index("Panel B")

    assert caption_pos > panel_a_pos, (
        f"Caption (pos {caption_pos}) must come after Panel A content (pos {panel_a_pos}) "
        "in SVG document order."
    )
    assert caption_pos > panel_b_pos, (
        f"Caption (pos {caption_pos}) must come after Panel B content (pos {panel_b_pos}) "
        "in SVG document order."
    )


# ---------------------------------------------------------------------------
# D10-T10: ConcatChart (general grid) — figure title/subtitle/caption each once
# ---------------------------------------------------------------------------


def test_d10_t10_concat_chart_figure_chrome_appears_once(two_charts):
    """ConcatChart.properties(title=, subtitle=, caption=) renders each text once.

    Previously ConcatChart silently dropped figure chrome: _figure_title was
    stored but never threaded into the (then-existing) grid compositor, so no
    chrome appeared.
    """
    from ferrum.composition import ConcatChart

    c1, c2 = two_charts
    composed = ConcatChart(c1, c2).properties(
        title="Concat Title",
        subtitle="Concat Subtitle",
        caption="Concat Caption",
    )
    svg = composed.to_svg()

    assert svg.count("Concat Title") == 1, (
        f"Expected figure title once in ConcatChart; got {svg.count('Concat Title')} copies."
    )
    assert svg.count("Concat Subtitle") == 1, (
        f"Expected figure subtitle once in ConcatChart; got {svg.count('Concat Subtitle')} copies."
    )
    assert svg.count("Concat Caption") == 1, (
        f"Expected figure caption once in ConcatChart; got {svg.count('Concat Caption')} copies."
    )


# ---------------------------------------------------------------------------
# D10-T11: ConcatChart — figure chrome survives .theme() / .configure() rebuild
# ---------------------------------------------------------------------------


def test_d10_t11_concat_chart_chrome_survives_rebuild(two_charts):
    """Figure chrome set via .properties() must survive a .theme() rebuild.

    _rebuild_with_charts previously returned a new ConcatChart without copying
    _figure_title/_figure_subtitle/_figure_caption, so a subsequent .theme()
    or .configure() call would silently drop any figure-level chrome.
    """
    from ferrum.composition import ConcatChart

    c1, c2 = two_charts
    composed = (
        ConcatChart(c1, c2)
        .properties(title="Survive Title", caption="Survive Caption")
        .theme(fm.themes.dark)
    )
    svg = composed.to_svg()

    assert "Survive Title" in svg, "Figure title was lost after .theme() rebuild on ConcatChart."
    assert "Survive Caption" in svg, (
        "Figure caption was lost after .theme() rebuild on ConcatChart."
    )


# ---------------------------------------------------------------------------
# D10-T12: empty dataset — caption is not silently dropped
# ---------------------------------------------------------------------------


def test_d10_t12_empty_dataset_caption_survives():
    """A chart with an empty dataset still renders the caption from .properties(caption=).

    Previously the empty-dataset fast-path in Chart.to_svg returned the
    placeholder SVG before reaching the caption post-wrap, so the caption
    was silently dropped.  The fix applies the same caption wrap to the
    empty-dataset branch.
    """
    df = pl.DataFrame({"x": pl.Series([], dtype=pl.Int64), "y": pl.Series([], dtype=pl.Int64)})
    chart = fm.Chart(df).mark_point().encode(x="x", y="y").properties(caption="Source: empty data")
    svg = chart.to_svg()

    assert "Source: empty data" in svg, (
        "Caption was silently dropped for an empty dataset. "
        "The empty-dataset fast-path must apply the caption wrap."
    )


def test_d10_t12b_empty_dataset_no_caption_path_unchanged():
    """An empty dataset without a caption renders the bare placeholder SVG unchanged.

    No caption wrap must be applied when no caption is set — this keeps the
    no-caption empty-dataset path byte-identical to its pre-fix form.
    """
    df = pl.DataFrame({"x": pl.Series([], dtype=pl.Int64), "y": pl.Series([], dtype=pl.Int64)})
    chart = fm.Chart(df).mark_point().encode(x="x", y="y")
    svg = chart.to_svg()

    assert "<!-- empty dataset -->" in svg, (
        "Empty-dataset placeholder comment must be present when no caption is set."
    )
    assert "<caption" not in svg.lower(), (
        "No caption element should appear when no caption was set."
    )


# ---------------------------------------------------------------------------
# D10-T13..T20: figure-chrome horizontal positioning (issue #1)
#
# Figure-level chrome (title / subtitle / caption) on composites and single-
# chart captions previously rendered flush-left at x=0, ignoring
# configure_padding(left=) and configure_title(anchor=).  The Rust emitter now
# honors a left/right inset and anchor; the Python layer resolves those from a
# chart's merged configure dict.  These tests assert the resolved positions on
# the rendered SVG.
# ---------------------------------------------------------------------------


def test_d10_t13_hconcat_default_chrome_inset(two_charts):
    """Default hconcat chrome sits at x=16, start-anchored (was x=0)."""
    c1, c2 = two_charts
    svg = (c1 | c2).properties(title="Figure Title", caption="Figure Caption").to_svg()

    title = _figure_title_node(svg, "Figure Title")
    assert title.get("x") == "16", f"default figure title x should be 16; got {title.get('x')!r}"
    assert title.get("text-anchor") == "start"

    caption = _find_text_node(svg, "Figure Caption")
    assert caption.get("x") == "16", f"default caption x should be 16; got {caption.get('x')!r}"
    assert caption.get("text-anchor") == "start"


def test_d10_t14_vconcat_default_chrome_inset(two_charts):
    """Default vconcat chrome sits at x=16, start-anchored."""
    c1, c2 = two_charts
    svg = (c1 & c2).properties(title="Figure Title", caption="Figure Caption").to_svg()

    title = _figure_title_node(svg, "Figure Title")
    assert title.get("x") == "16"
    assert title.get("text-anchor") == "start"

    caption = _find_text_node(svg, "Figure Caption")
    assert caption.get("x") == "16"
    assert caption.get("text-anchor") == "start"


def test_d10_t15_concat_grid_default_chrome_inset(two_charts):
    """Default grid-composite (ConcatChart) chrome sits at x=16, start-anchored."""
    from ferrum.composition import ConcatChart

    c1, c2 = two_charts
    svg = ConcatChart(c1, c2).properties(title="Figure Title", caption="Figure Caption").to_svg()

    title = _figure_title_node(svg, "Figure Title")
    assert title.get("x") == "16"
    assert title.get("text-anchor") == "start"

    caption = _find_text_node(svg, "Figure Caption")
    assert caption.get("x") == "16"
    assert caption.get("text-anchor") == "start"


def test_d10_t16_padding_left_shifts_chrome(two_charts):
    """configure_padding(left=60) moves title and caption to x=60 (not dropped)."""
    c1, c2 = two_charts
    svg = (
        (c1 | c2)
        .properties(title="Figure Title", caption="Figure Caption")
        .configure_padding(left=60, auto=False)
        .to_svg()
    )

    title = _figure_title_node(svg, "Figure Title")
    assert title.get("x") == "60", (
        f"configure_padding(left=60) must shift the figure title to x=60; got {title.get('x')!r}"
    )
    assert title.get("text-anchor") == "start"

    caption = _find_text_node(svg, "Figure Caption")
    assert caption.get("x") == "60", (
        f"configure_padding(left=60) must shift the caption to x=60; got {caption.get('x')!r}"
    )


def test_d10_t17_anchor_middle_centers_chrome(two_charts):
    """configure_title(anchor='middle') centers chrome at width/2, middle-anchored."""
    c1, c2 = two_charts
    svg = (
        (c1 | c2)
        .properties(title="Figure Title", caption="Figure Caption")
        .configure_title(anchor="middle")
        .to_svg()
    )
    half = _root_width(svg) / 2

    title = _figure_title_node(svg, "Figure Title")
    assert title.get("text-anchor") == "middle"
    assert float(title.get("x")) == pytest.approx(half), (
        f"anchor=middle must center the figure title at width/2={half}; got x={title.get('x')!r}"
    )

    caption = _find_text_node(svg, "Figure Caption")
    assert caption.get("text-anchor") == "middle"
    assert float(caption.get("x")) == pytest.approx(half)


def test_d10_t18_anchor_end_right_aligns_chrome(two_charts):
    """configure_title(anchor='end') right-aligns the figure title."""
    c1, c2 = two_charts
    svg = (c1 | c2).properties(title="Figure Title").configure_title(anchor="end").to_svg()

    title = _figure_title_node(svg, "Figure Title")
    assert title.get("text-anchor") == "end", (
        f"anchor=end must right-align the figure title; got {title.get('text-anchor')!r}"
    )


def test_d10_t19_single_chart_caption_default_and_padding():
    """Single-chart caption defaults to x=16 and honors configure_padding(left=)."""
    df = pl.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})

    default_svg = fm.Chart(df).mark_point().encode(x="x", y="y").properties(caption="Note").to_svg()
    caption = _find_text_node(default_svg, "Note")
    assert caption.get("x") == "16", (
        f"single-chart caption should default to x=16; got {caption.get('x')!r}"
    )
    assert caption.get("text-anchor") == "start"

    padded_svg = (
        fm.Chart(df)
        .mark_point()
        .encode(x="x", y="y")
        .properties(caption="Note")
        .configure_padding(left=40, auto=False)
        .to_svg()
    )
    caption = _find_text_node(padded_svg, "Note")
    assert caption.get("x") == "40", (
        f"configure_padding(left=40) must shift the single-chart caption to x=40; "
        f"got {caption.get('x')!r}"
    )


def test_d10_t20_composite_without_chrome_unaffected(two_charts):
    """A composite with no figure title/caption renders no figure-chrome text node.

    Wiring the inset/anchor resolution must not introduce a chrome band when no
    figure title, subtitle, or caption was set.
    """
    c1, c2 = two_charts
    svg = (c1 | c2).configure_padding(left=60, auto=False).to_svg()

    root = ET.fromstring(svg)
    figure_titles = [
        el
        for el in root.iter(f"{_SVG_NS}text")
        if el.get("font-size") == "16" and el.get("font-weight") == "600"
    ]
    assert figure_titles == [], (
        "No figure-level title band should be emitted when no figure chrome is set."
    )


# ---------------------------------------------------------------------------
# D10-T21..T24: additional sibling-path regression tests for chrome positioning
#
# These cover paths that T13-T20 do not exercise:
#   T21 — empty-dataset single-chart caption (distinct render branch in _render.py)
#   T22 — figure subtitle shares the resolved anchor (separate emitter line)
#   T23 — anchor=end with a custom right_inset (T18 only checked anchor, not inset)
#   T24 — configure_padding(left=0) is a falsy-but-set value that must reach Rust
# ---------------------------------------------------------------------------


def test_d10_t21_empty_dataset_caption_default_inset():
    """Regression: empty-dataset single-chart caption rendered flush-left at x=0 (separate render path).

    The empty-data fast-path in _render.py (~line 662) is a distinct
    wrap_svg_with_chrome call from the normal path.  It must also honour the
    default left inset (16) and start-anchor, not emit the caption at x=0.
    """
    empty = pl.DataFrame({"x": [], "y": []}, schema={"x": pl.Float64, "y": pl.Float64})
    svg = fm.Chart(empty).mark_point().encode(x="x", y="y").properties(caption="EmptyCap").to_svg()

    caption = _find_text_node(svg, "EmptyCap")
    assert caption.get("x") == "16", (
        f"empty-dataset caption x should default to 16; got {caption.get('x')!r}"
    )
    assert caption.get("text-anchor") == "start", (
        f"empty-dataset caption text-anchor should be 'start'; got {caption.get('text-anchor')!r}"
    )


def test_d10_t22_subtitle_honors_anchor(two_charts):
    """Regression: figure subtitle must share the resolved anchor from configure_title.

    The subtitle is emitted on a separate line from the title in the Rust chrome
    emitter.  With anchor=middle it must be centered at width/2, not left-aligned.
    (The subtitle node has font-size 13, so _figure_title_node is not used.)
    """
    c1, c2 = two_charts
    svg = (
        (c1 | c2).properties(title="Main", subtitle="Sub").configure_title(anchor="middle").to_svg()
    )
    half = _root_width(svg) / 2

    subtitle = _find_text_node(svg, "Sub")
    assert subtitle.get("text-anchor") == "middle", (
        f"subtitle text-anchor should be 'middle' with anchor=middle; "
        f"got {subtitle.get('text-anchor')!r}"
    )
    assert float(subtitle.get("x")) == pytest.approx(half), (
        f"subtitle x should be width/2={half}; got {subtitle.get('x')!r}"
    )


def test_d10_t23_anchor_end_with_custom_right_inset(two_charts):
    """Regression: configure_padding(right=50) + anchor=end must place title at width-50.

    T18 only checked that anchor=end produces text-anchor='end' without a custom
    right inset.  This guards the right_inset field specifically: the title x
    must be root_width - 50 when a non-default right padding is configured.
    """
    c1, c2 = two_charts
    svg = (
        (c1 | c2)
        .properties(title="T")
        .configure_padding(right=50, auto=False)
        .configure_title(anchor="end")
        .to_svg()
    )
    expected_x = _root_width(svg) - 50

    title = _figure_title_node(svg, "T")
    assert title.get("text-anchor") == "end", (
        f"anchor=end must right-align the figure title; got {title.get('text-anchor')!r}"
    )
    assert float(title.get("x")) == pytest.approx(expected_x), (
        f"figure title x should be width-50={expected_x}; got {title.get('x')!r}"
    )


def test_d10_t24_padding_left_zero_passes_through(two_charts):
    """Regression: configure_padding(left=0) must reach the emitter as x=0, not be dropped to the default 16.

    The chrome_kwargs resolver uses ``is not None`` to decide whether to forward
    left_inset to Rust.  A value of 0 is falsy but is a legitimate explicit
    setting and must not be silently omitted, which would let the Rust default
    (16) override the user's choice.
    """
    c1, c2 = two_charts
    svg = (c1 | c2).properties(title="T").configure_padding(left=0, auto=False).to_svg()

    title = _figure_title_node(svg, "T")
    assert title.get("x") == "0", (
        f"configure_padding(left=0) must forward x=0 to Rust, not fall back to default 16; "
        f"got {title.get('x')!r}"
    )
