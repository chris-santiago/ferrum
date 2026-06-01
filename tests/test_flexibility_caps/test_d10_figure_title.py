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

Rust layer: ``compose_svg_vertical`` / ``compose_svg_horizontal``
(``crates/ferrum-core/src/render/compositor.rs:302 / :262``) accept only SVG
strings and a spacing value; they have no facility to inject a figure-wide
title band before the panel grid or a caption band below it.

``compose_svg_grid``
(``crates/ferrum-core/src/render/grid_compose.rs:42``) similarly has no
figure-level title or caption parameter.

For a fix to land, both layers must change:
  - Python: ``_ChartLike`` needs a ``figure_title`` / ``figure_subtitle`` /
    ``figure_caption`` store (or an overloaded ``properties`` that distinguishes
    figure-level from per-panel kwargs), and the render call must pass those
    values through.
  - Rust: ``compose_svg_vertical``, ``compose_svg_horizontal``, and
    ``compose_svg_grid`` need an optional title-band (above) and caption-band
    (below) parameter so that a single text node wraps the whole output SVG.

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

import polars as pl
import pytest

import ferrum as fm


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
    svg = composed.show_svg()

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
    svg = composed.show_svg()

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
    svg = composed.show_svg()

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
    svg = composed.show_svg()

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
    svg = composed.show_svg()

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
    svg = result.show_svg()
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
    svg = result.show_svg()
    assert "A caption note" in svg, "caption text must appear in the rendered SVG"


# ---------------------------------------------------------------------------
# D10-T8: figure title appears above all panels (document order)
# ---------------------------------------------------------------------------


def test_d10_t8_figure_title_appears_before_panel_content(two_charts_with_panel_titles):
    """The figure-level title must precede all panel body content in SVG order.

    EXPECTED TO FAIL (TDD RED): currently the title is per-child, so the first
    child's title appears at its natural position inside that child's SVG, not
    before the entire composed figure.

    We verify document order: the figure title text node must appear at an
    earlier byte offset than both per-panel titles.
    """
    c1, c2 = two_charts_with_panel_titles
    composed = (c1 & c2).properties(title="Figure Title")
    svg = composed.show_svg()

    assert "Figure Title" in svg, "Figure Title must be present"
    assert "Panel A" in svg, "Panel A must be present"
    assert "Panel B" in svg, "Panel B must be present"

    figure_pos = svg.index("Figure Title")
    panel_a_pos = svg.index("Panel A")
    panel_b_pos = svg.index("Panel B")

    assert figure_pos < panel_a_pos, (
        f"Figure title (pos {figure_pos}) must precede Panel A title (pos {panel_a_pos}) "
        "in SVG document order."
    )
    assert figure_pos < panel_b_pos, (
        f"Figure title (pos {figure_pos}) must precede Panel B title (pos {panel_b_pos}) "
        "in SVG document order."
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
    svg = composed.show_svg()

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
    stored but never threaded into compose_svg_grid, so no chrome appeared.
    """
    from ferrum.composition import ConcatChart

    c1, c2 = two_charts
    composed = ConcatChart(c1, c2).properties(
        title="Concat Title",
        subtitle="Concat Subtitle",
        caption="Concat Caption",
    )
    svg = composed.show_svg()

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
    svg = composed.show_svg()

    assert "Survive Title" in svg, "Figure title was lost after .theme() rebuild on ConcatChart."
    assert "Survive Caption" in svg, (
        "Figure caption was lost after .theme() rebuild on ConcatChart."
    )


# ---------------------------------------------------------------------------
# D10-T12: empty dataset — caption is not silently dropped
# ---------------------------------------------------------------------------


def test_d10_t12_empty_dataset_caption_survives():
    """A chart with an empty dataset still renders the caption from .properties(caption=).

    Previously the empty-dataset fast-path in Chart.show_svg returned the
    placeholder SVG before reaching the caption post-wrap, so the caption
    was silently dropped.  The fix applies the same caption wrap to the
    empty-dataset branch.
    """
    df = pl.DataFrame({"x": pl.Series([], dtype=pl.Int64), "y": pl.Series([], dtype=pl.Int64)})
    chart = fm.Chart(df).mark_point().encode(x="x", y="y").properties(caption="Source: empty data")
    svg = chart.show_svg()

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
    svg = chart.show_svg()

    assert "<!-- empty dataset -->" in svg, (
        "Empty-dataset placeholder comment must be present when no caption is set."
    )
    assert "<caption" not in svg.lower(), (
        "No caption element should appear when no caption was set."
    )
