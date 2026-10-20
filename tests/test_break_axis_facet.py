"""``BreakAxis`` on a faceted chart: every panel, one voice (#230-7, #231).

Before this, `build_scene` gated the whole structural build on
``panel_idx == 0``, so a faceted chart broke panel 0 and left panels 1..N on
the unbroken scale: two panels sharing ONE y axis, rendering the same values
at different heights, with no warning to say so. These pins cover the four
halves of the fix.

* **Geometry** — identical data in two panels places identically, and the
  gap-spanning bar clips into per-segment pieces in *both* panels.
* **Chrome** — each panel draws its own break indicator, inside its own rect.
* **Warning cadence** — breaks are built per panel, but the facts they report
  are not per-panel facts. "This break removed nothing" is a function of
  ``(spec, resolved domain)``, which a shared-scale facet resolves once, so it
  is reported once. The one shape where it genuinely varies is
  ``share_scale(y="independent")``: there the single report names the panel it
  was measured on and how many panels share it, rather than going silent on
  the panels the gap missed or repeating itself once per panel.
* **Annotations** — chart-level annotations are built in ONE place (against the
  reference panel's scales), so their in-gap suppression cannot diverge per
  panel. A composite's leaves are separate scenes and keep their own voice: an
  hconcat of two leaves whose annotations both drop warns twice, once per leaf.
"""

from __future__ import annotations

import re
import warnings

import polars as pl

import ferrum as fm
import ferrum.annotation as ann
from _svg_rects import fill_rects

_MARK_FILL = "#2563eb"


def _bars(svg: str) -> list[tuple[float, float, float, float]]:
    """``(x, y, w, h)`` for every mark rect painted in the default bar fill.

    Binds to ``tests/_svg_rects.py``'s ``fill_rects`` (the home for this
    fill-filtered rect extraction, since it was identically duplicated across
    this module, ``test_break_axis_geometry.py`` and
    ``test_break_axis_coordflip.py``).
    """
    return fill_rects(svg, _MARK_FILL)


def _panel_split(svg: str) -> tuple[list, list]:
    """Bars split into the left and right facet panel by their x midpoint."""
    bars = _bars(svg)
    assert bars, "the fixture must paint bars"
    mid = (min(b[0] for b in bars) + max(b[0] + b[2] for b in bars)) / 2
    return (
        [b for b in bars if b[0] + b[2] / 2 < mid],
        [b for b in bars if b[0] + b[2] / 2 >= mid],
    )


def _diagonals(svg: str) -> list[tuple[float, float, float, float]]:
    """Every ``<line>`` that is neither horizontal nor vertical.

    The slash break indicator is the only diagonal stroke these charts emit,
    so this isolates it from gridlines, axis domain lines, and tick dashes.
    """
    return [
        tuple(float(g) for g in m.groups())
        for m in re.finditer(
            r'<line x1="([-\d.]+)" y1="([-\d.]+)" x2="([-\d.]+)" y2="([-\d.]+)"', svg
        )
        if m.group(1) != m.group(3) and m.group(2) != m.group(4)
    ]


def _two_panel_df(q_high: float = 900.0) -> pl.DataFrame:
    """Two facet panels carrying the same two categories.

    ``q_high`` is the second panel's high value: equal to the first panel's
    (``900``) the two panels report the same resolved domain. Far below it,
    the two panels' bar-chart zero-baseline domains differ only under
    `share_scale(y="independent")`; the default shared resolve gives every
    panel the one ``(0, 900)`` domain the shared axis shows.
    """
    return pl.DataFrame(
        {
            "c": ["A", "B", "A", "B"],
            "v": [10.0, 900.0, 10.0, q_high],
            "g": ["p", "p", "q", "q"],
        }
    )


def _render(chart) -> tuple[str, list[str]]:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        svg = chart.to_svg()
    return svg, [str(w.message) for w in caught]


# --------------------------------------------------------------------------
# geometry
# --------------------------------------------------------------------------


def test_break_applies_to_every_facet_panel():
    """The measured panel-1 divergence, flipped.

    Pre-fix: panel 0's 900-bar clipped into two pieces while panel 1's was
    drawn full height on the unbroken scale — the same value at two different
    positions under one shared axis.
    """
    chart = fm.Chart(_two_panel_df()).mark_bar().encode(x="c:N", y="v:Q").facet(
        col="g"
    ) + fm.BreakAxis(axis="y", gap=(150, 850))
    svg, msgs = _render(chart)

    left, right = _panel_split(svg)
    assert [(b[1], b[3]) for b in left] == [(b[1], b[3]) for b in right], (
        "identical data in both panels must land at identical (y, height); "
        f"got left={left} right={right}"
    )
    assert len(left) == 3, (
        f"the gap-spanning bar clips into two pieces beside the short one: {left}"
    )
    assert not msgs, f"an effective break on a linear shared-scale facet is silent: {msgs}"


def test_unbroken_faceted_chart_draws_one_rect_per_row():
    """The discriminator for the test above: without the break, each panel
    paints one whole bar per row, so a 3-vs-2 rect count really is the clip."""
    chart = fm.Chart(_two_panel_df()).mark_bar().encode(x="c:N", y="v:Q").facet(col="g")
    left, right = _panel_split(chart.to_svg())
    assert (len(left), len(right)) == (2, 2)


def test_ineffective_gap_leaves_a_faceted_chart_byte_identical_to_unbroken():
    """The C1 invariant, extended across panels: a break whose gaps miss the
    domain renders exactly the unbroken chart on EVERY panel, not just the
    first."""
    base = fm.Chart(_two_panel_df()).mark_bar().encode(x="c:N", y="v:Q").facet(col="g")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        broken = (base + fm.BreakAxis(axis="y", gap=(2000, 3000))).to_svg()
    assert broken == base.to_svg()


# --------------------------------------------------------------------------
# indicators
# --------------------------------------------------------------------------


def test_break_indicator_renders_in_every_facet_panel():
    """Each panel marks its own cut. A broken scale with no indicator on it is
    a chart that lies about its own axis."""
    chart = fm.Chart(_two_panel_df()).mark_bar().encode(x="c:N", y="v:Q").facet(
        col="g"
    ) + fm.BreakAxis(axis="y", gap=(150, 850))
    svg, _ = _render(chart)

    left, right = _panel_split(svg)
    left_span = (min(b[0] for b in left), max(b[0] + b[2] for b in left))
    right_span = (min(b[0] for b in right), max(b[0] + b[2] for b in right))

    slashes = _diagonals(svg)
    in_left = [s for s in slashes if left_span[0] <= s[0] <= left_span[1]]
    in_right = [s for s in slashes if right_span[0] <= s[0] <= right_span[1]]
    assert in_left, f"no break indicator inside the left panel: {slashes}"
    assert in_right, f"no break indicator inside the right panel: {slashes}"
    assert len(in_left) == len(in_right), (
        f"both panels draw the same indicator: left={in_left} right={in_right}"
    )
    # Same vertical position in both panels (one shared y axis), different
    # horizontal one (each panel's own rect).
    assert [s[1] for s in in_left] == [s[1] for s in in_right]
    assert {s[0] for s in in_left}.isdisjoint({s[0] for s in in_right})


# --------------------------------------------------------------------------
# warning cadence
# --------------------------------------------------------------------------


def _ineffective(msgs: list[str]) -> list[str]:
    return [m for m in msgs if "had no effect" in m]


def test_ineffective_break_on_a_shared_scale_facet_warns_once_not_once_per_panel():
    """A shared-scale facet resolves ONE domain, so "no gap overlaps it" is one
    fact about the chart however many panels observe it."""
    chart = fm.Chart(_two_panel_df()).mark_bar().encode(x="c:N", y="v:Q").facet(
        col="g"
    ) + fm.BreakAxis(axis="y", gap=(2000, 3000))
    _, msgs = _render(chart)
    reports = _ineffective(msgs)
    assert len(reports) == 1, f"one fact, one report: {msgs}"
    assert "panel" not in reports[0], (
        "a fact that held on every panel reads exactly as it does on a "
        f"single-panel chart: {reports[0]}"
    )


def test_free_scale_facet_names_the_panels_a_break_missed():
    """`share_scale(y="independent")` is the one shape where effectiveness
    genuinely varies by panel. Going silent on the panels the gap missed would
    be dishonest and repeating the warning per panel would be noise, so the one
    report names the panel it was measured on and the fraction that share it."""
    chart = fm.Chart(_two_panel_df(q_high=20.0)).mark_bar().encode(x="c:N", y="v:Q").facet(
        col="g"
    ).share_scale(y="independent") + fm.BreakAxis(axis="y", gap=(150, 850))
    svg, msgs = _render(chart)
    reports = _ineffective(msgs)
    assert len(reports) == 1, f"still ONE report, not one per panel: {msgs}"
    assert "facet panel 1" in reports[0], f"the report names the panel: {reports[0]}"
    assert "1 of 2 panels" in reports[0], (
        f"the report says the fact held on only some panels: {reports[0]}"
    )

    # And the panel the gap DID overlap still breaks: a partial report is a
    # qualification, not a suppression.
    left, right = _panel_split(svg)
    assert len(left) == 3, f"panel 0's gap-spanning bar still clips: {left}"
    assert len(right) == 2, f"panel 1's domain stops below the gap: {right}"


def test_free_scale_facets_nonlinear_warning_stays_flat_beside_a_qualified_ineffective_one():
    """`BreakAxisOnNonlinearScale` deliberately does NOT get the same partial-
    panel qualification `BreakAxisIneffective` gets: it is a claim about the
    SPEC (a log axis's inverse is not the break's linear reverse map), equally
    true on every panel the break reaches, so naming a subset would add
    nothing actionable. A free-scale facet with a log y and a gap effective on
    only one panel emits a FLAT nonlinear warning (no panel named) beside a
    QUALIFIED ineffective warning for the panel the gap missed."""
    chart = fm.Chart(_two_panel_df(q_high=20.0)).mark_bar().encode(
        x="c:N", y=fm.Y("v:Q", scale={"type": "log"})
    ).facet(col="g").share_scale(y="independent") + fm.BreakAxis(axis="y", gap=(150, 850))
    _, msgs = _render(chart)

    nonlinear = [m for m in msgs if "targets a log scale" in m]
    assert len(nonlinear) == 1, f"only one panel's domain overlaps the gap: {msgs}"
    assert "panel" not in nonlinear[0], (
        f"the nonlinear warning reports flat even though only 1 of 2 panels "
        f"observed it: {nonlinear[0]}"
    )

    ineffective = _ineffective(msgs)
    assert len(ineffective) == 1, f"one panel missed the gap entirely: {msgs}"
    assert "facet panel 1" in ineffective[0] and "1 of 2 panels" in ineffective[0], (
        f"the ineffective report, unlike the nonlinear one beside it, IS "
        f"qualified: {ineffective[0]}"
    )


def test_free_scale_facet_names_the_panel_when_a_gap_misses_every_panel_with_diverging_domains():
    """The all-report divergence hole (T11 cycle-4 quality review, finding 1).

    The pre-fix qualifier fired only when the reporting count fell short of
    the panel count. Here every panel reports `NoOverlap` — the gap misses
    both panels' domains entirely — but under `share_scale(y="independent")`
    the two panels' domains genuinely differ (bar's zero-baseline default
    widens them to ``(0, 900)`` and ``(0, 30)``), so the flushed payload was
    panel 0's domain, unattributed, presented as if it were the chart's."""
    chart = fm.Chart(_two_panel_df(q_high=30.0)).mark_bar().encode(x="c:N", y="v:Q").facet(
        col="g"
    ).share_scale(y="independent") + fm.BreakAxis(axis="y", gap=(2000, 3000))
    _, msgs = _render(chart)
    reports = _ineffective(msgs)
    assert len(reports) == 1, f"still ONE report, not one per panel: {msgs}"
    reason = reports[0]
    assert "(0, 900)" in reason, (
        f"the report still names the first-reporting panel's own domain: {reason}"
    )
    assert "facet panel 0" in reason, (
        f"unlike the pre-fix flush, the payload is attributed to the panel it "
        f"was measured on rather than presented as the chart's: {reason}"
    )
    assert "applies on the rest" not in reason, (
        f"every panel missed its own gap here, so there is no 'rest' the break applied on: {reason}"
    )


def test_default_resolve_bar_facet_reports_one_unqualified_shared_domain():
    """A bar facet under the default (shared) y resolve resolves one domain.

    ``mark_bar()`` writes a domain-less zero scale on the value channel, and
    under a shared facet axis each panel's scale spans the rows of every
    panel, the domain the shared axis shows. Groups with different maxima
    (``900`` and ``30``) therefore meet one ``(0, 900)`` domain, and a gap
    that misses it is one chart-level fact: reported once, unqualified,
    naming that shared domain and no panel."""
    chart = fm.Chart(_two_panel_df(q_high=30.0)).mark_bar().encode(x="c:N", y="v:Q").facet(
        col="g"
    ) + fm.BreakAxis(axis="y", gap=(2000, 3000))
    _, msgs = _render(chart)
    assert _ineffective(msgs) == [
        "BreakAxis on the y axis had no effect: no gap overlaps the axis domain (0, 900)"
    ], f"one unqualified report naming the shared domain: {msgs}"


def test_row_faceted_break_places_every_panel_at_its_own_vertical_offset():
    """The pin gap the T11 cycle-4 quality review named (finding 5): every
    other pin here uses a column-wrap facet, where every panel shares one
    y pixel range, so only x discriminates the per-panel plot_area/scale
    pairing. `.facet(row=...)` stacks panels vertically instead, so panel 1
    occupies a genuinely different y extent than panel 0 -- the shape a
    column facet cannot exercise."""
    chart = fm.Chart(_two_panel_df()).mark_bar().encode(x="c:N", y="v:Q").facet(
        row="g"
    ) + fm.BreakAxis(axis="y", gap=(150, 850))
    svg, msgs = _render(chart)

    bars = _bars(svg)
    assert bars, "the fixture must paint bars"
    mid = (min(b[1] for b in bars) + max(b[1] + b[3] for b in bars)) / 2
    top = [b for b in bars if b[1] + b[3] / 2 < mid]
    bottom = [b for b in bars if b[1] + b[3] / 2 >= mid]
    assert top and bottom, f"a row facet must stack panels at distinct y extents: {bars}"
    assert max(b[1] + b[3] for b in top) <= min(b[1] for b in bottom), (
        f"the two panels' bars must not overlap in y: top={top} bottom={bottom}"
    )

    # Each panel's own break indicator must land within that panel's own
    # vertical span -- reusing the reference panel's y for both panels would
    # place panel 1's indicator inside panel 0's span instead.
    top_span = (min(b[1] for b in top), max(b[1] + b[3] for b in top))
    bottom_span = (min(b[1] for b in bottom), max(b[1] + b[3] for b in bottom))
    slashes = _diagonals(svg)
    in_top = [s for s in slashes if top_span[0] <= s[1] <= top_span[1]]
    in_bottom = [s for s in slashes if bottom_span[0] <= s[1] <= bottom_span[1]]
    assert in_top, f"no break indicator inside the top panel: {slashes}"
    assert in_bottom, f"no break indicator inside the bottom panel: {slashes}"

    # Identical data breaks identically once each panel's own vertical
    # offset is subtracted out.
    top_rel = sorted((round(b[1] - top_span[0], 1), round(b[3], 1)) for b in top)
    bottom_rel = sorted((round(b[1] - bottom_span[0], 1), round(b[3], 1)) for b in bottom)
    assert top_rel == bottom_rel, (
        f"identical data must break identically relative to each panel's own "
        f"y offset: top={top_rel} bottom={bottom_rel}"
    )
    assert len(top) == 3, f"the gap-spanning bar still clips under a row facet: {top}"
    assert not msgs, f"an effective break on a linear shared-scale row facet is silent: {msgs}"


def test_three_panel_free_scale_facet_names_the_first_missed_panel_and_the_full_fraction():
    """The `>2`-panel message shape: `PanelFact` keeps only the FIRST
    reporting panel's payload, so a three-panel independent-scale facet where
    the gap misses two of the three panels names one domain and a fraction
    counting every panel that shares the fact, not a per-panel list."""
    df = pl.DataFrame(
        {
            "c": ["A", "B", "A", "B", "A", "B"],
            "v": [10.0, 900.0, 10.0, 20.0, 10.0, 30.0],
            "g": ["p", "p", "q", "q", "r", "r"],
        }
    )
    chart = fm.Chart(df).mark_bar().encode(x="c:N", y="v:Q").facet(col="g").share_scale(
        y="independent"
    ) + fm.BreakAxis(axis="y", gap=(150, 850))
    _, msgs = _render(chart)
    reports = _ineffective(msgs)
    assert len(reports) == 1, f"two panels sharing the missed fact report it ONCE: {msgs}"
    assert "facet panel 1" in reports[0], (
        f"the report names the first panel it was measured on: {reports[0]}"
    )
    assert "2 of 3 panels" in reports[0], (
        f"the fraction counts every panel that shares the fact: {reports[0]}"
    )


def test_merge_conflict_on_a_faceted_chart_warns_once_not_once_per_panel():
    """Two same-axis `BreakAxis` specs disagreeing on `break_style` is a fact
    about the SPEC, independent of any panel."""
    chart = (
        fm.Chart(_two_panel_df()).mark_bar().encode(x="c:N", y="v:Q").facet(col="g")
        + fm.BreakAxis(axis="y", gap=(150, 400), break_style="slash")
        + fm.BreakAxis(axis="y", gap=(500, 850), break_style="zigzag")
    )
    _, msgs = _render(chart)
    conflicts = [m for m in msgs if "disagree on break_style" in m]
    assert len(conflicts) == 1, f"one conflict, one report: {msgs}"


# --------------------------------------------------------------------------
# annotations (#231's fourth coverage gap)
# --------------------------------------------------------------------------


def test_in_gap_annotation_on_a_faceted_broken_chart_is_suppressed_once():
    """Chart-level annotations are built in ONE place — against the reference
    panel's scales — so their data-space suppression cannot diverge per panel.
    The span lies wholly inside the gap: it renders nowhere, and says so once.
    """
    chart = (
        fm.Chart(_two_panel_df()).mark_bar().encode(x="c:N", y="v:Q").facet(col="g")
        + fm.BreakAxis(axis="y", gap=(150, 850))
        + fm.Annotate(ann.span("y", 300.0, 700.0, fill="#eeeeee"))
    )
    svg, msgs = _render(chart)
    assert 'fill="#eeeeee"' not in svg, "the suppressed span must not render on ANY panel"
    suppressed = [m for m in msgs if "suppressed" in m and "y axis" in m]
    assert len(suppressed) == 1, f"one annotation, one report: {msgs}"


def test_surviving_annotation_on_a_faceted_broken_chart_rides_the_break():
    """The other half: an out-of-gap annotation is not suppressed, and it is
    placed on the BROKEN scale (not left at its unbroken position)."""
    base = fm.Chart(_two_panel_df()).mark_bar().encode(x="c:N", y="v:Q").facet(col="g")
    span = fm.Annotate(ann.span("y", 20.0, 60.0, fill="#eeeeee"))
    broken, msgs = _render(base + fm.BreakAxis(axis="y", gap=(150, 850)) + span)
    unbroken, _ = _render(base + span)

    def band(svg):
        return re.findall(
            r'<rect x="[-\d.]+" y="([-\d.]+)" width="[\d.]+" height="([\d.]+)" '
            r'fill="#eeeeee"',
            svg,
        )

    assert not msgs, f"an out-of-gap annotation is not suppressed: {msgs}"
    assert len(band(broken)) == 1, "the chart-level annotation is drawn once"
    assert band(broken) != band(unbroken), (
        "the surviving annotation must be re-placed on the compressed scale"
    )


def test_each_composite_leaf_reports_its_own_suppressed_annotation():
    """Composite leaves are separate scenes and keep their own voice.

    The scene-level report cadence deduplicates WITHIN one scene; it must not
    reach across leaves, or an hconcat of two independently-broken charts would
    silently report only one of them. Two leaves, two reports.
    """
    df = pl.DataFrame({"c": ["A", "B"], "v": [10.0, 900.0]})

    def leaf():
        return (
            fm.Chart(df).mark_bar().encode(x="c:N", y="v:Q")
            + fm.BreakAxis(axis="y", gap=(150, 850))
            + fm.Annotate(ann.span("y", 300.0, 700.0, fill="#eeeeee"))
        )

    svg, msgs = _render(leaf() | leaf())
    assert 'fill="#eeeeee"' not in svg
    suppressed = [m for m in msgs if "suppressed" in m and "y axis" in m]
    assert len(suppressed) == 2, f"one report per leaf, not one per composite: {msgs}"
