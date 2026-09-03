"""Render-level pins for the legend orient/direction/values/cascade contract.

Batch B task 7 (spec ``.claude/output/specs/2026-09-02-batch-b-config-plumbing-design.md``
§4.4; decisions D5/D6/D7 + the NF-B13 adjudication). Every assertion here goes
through the real public API down to ``to_svg()`` — the legend contract is a
*rendered* contract, and the findings it closes (F-L04-04, F-L04-05, F-L07-02,
NF-B10, NF-B13) were all "the kwarg is accepted and the SVG does not change".

What is pinned:

- **orient × direction (all 8).** ``orient`` places the legend strip on a chart
  edge; ``direction`` arranges entries within it. Every combination draws every
  entry and raises no overflow warning at an adequate viewport. Before the fix
  ``estimate_legend_size`` sized off ``orient`` while ``layout_legend`` placed
  off ``direction``, so ``orient="right"`` + ``direction="horizontal"`` reserved
  a one-entry-wide column and dropped all three entries (F-L07-02/NF-B10).
- **Horizontal colorbar.** ``direction="horizontal"`` renders a left→right
  gradient with the ticks beneath it (D5).
- **``values`` on a categorical legend.** Filters and orders the entries,
  mirroring the colorbar arm; unknown names warn and are skipped (D6/F-L04-05).
- **``X(legend=)`` / ``Y(legend=)``.** The positional channels' documented
  ``legend=`` kwarg reaches the same per-channel override path ``Color``'s does
  (NF-B13, user-adjudicated: implement).
- **``orient="none"`` per channel.** Suppresses that channel's legend, matching
  the chart-level spelling.
- **Cascade (D7).** Per-channel beats chart-level on ``orient``/``columns``/
  ``title_font_size``; ``configure_legend(label_font_size=)`` sizes legend
  labels only and no longer resizes axis tick labels.
- **Byte-identity.** Charts that name none of the touched fields, and charts
  that name them redundantly (the value the default already had), render
  byte-identically.
- **Token validation.** One shared ``orient``/``direction`` vocabulary check
  refuses an unrecognized token the same way on every Python surface
  (``fm.Legend``, ``configure_legend``/``LegendConfig``, and a raw
  ``legend={...}`` dict) — the one exception being pure-construction refusal
  pins, which do not need to reach ``to_svg()`` to prove the boundary is loud.
"""

from __future__ import annotations

import re
import warnings

import polars as pl
import pytest

import ferrum as fm


# ---------------------------------------------------------------------------
# Fixtures + probes
# ---------------------------------------------------------------------------

#: Wide enough that no orient×direction combination is size-constrained: the
#: "no entries dropped" assertions must fail on the sizing BUG, never on a
#: genuinely too-small chart.
WIDE = {"width": 900, "height": 600}

CATEGORIES = ["alpha", "beta", "gamma"]


@pytest.fixture
def cat_df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "y": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "cat": CATEGORIES * 2,
        }
    )


@pytest.fixture
def num_df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "x": [1.0, 2.0, 3.0, 4.0, 5.0],
            "y": [1.0, 2.0, 3.0, 4.0, 5.0],
            "val": [0.0, 25.0, 50.0, 75.0, 100.0],
        }
    )


def texts(svg: str) -> list[str]:
    """All ``<text>`` node contents in document order."""
    return re.findall(r"<text[^>]*>([^<]+)</text>", svg)


def render(chart) -> tuple[str, list[str]]:
    """``(svg, warning_messages)`` for one chart, with warnings captured."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        svg = chart.to_svg()
    return svg, [str(w.message) for w in caught]


# ---------------------------------------------------------------------------
# D5 — orient x direction, all 8 combinations
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("orient", ["right", "left", "top", "bottom"])
@pytest.mark.parametrize("direction", ["vertical", "horizontal"])
def test_every_orient_direction_combination_draws_every_entry(cat_df, orient, direction):
    """All 8 orient×direction combinations keep every legend entry.

    RED before D5 for the four ``direction`` values that disagree with the
    orient-implied default (right/left + horizontal, top/bottom + vertical):
    the rect was sized for the orient's shape and the entries placed for the
    direction's, so the placement loop's ``max_n``/``max_rows`` floor-divided
    to a value smaller than the entry count and the surplus was dropped.
    """
    chart = (
        fm.Chart(cat_df)
        .mark_point()
        .encode(x="x:Q", y="y:Q", color="cat:N")
        .theme(fm.Theme(legend_direction=direction))
        .configure_legend(orient=orient)
        .properties(**WIDE)
    )
    svg, msgs = render(chart)
    drawn = texts(svg)
    for label in CATEGORIES:
        assert label in drawn, (
            f"orient={orient} direction={direction}: legend entry {label!r} was dropped; "
            f"drawn text nodes = {drawn}"
        )
    assert not [m for m in msgs if "legend overflowed" in m], (
        f"orient={orient} direction={direction} overflowed at {WIDE}: {msgs}"
    )


@pytest.mark.parametrize("orient", ["right", "left", "top", "bottom"])
@pytest.mark.parametrize("direction", ["vertical", "horizontal"])
def test_every_orient_direction_combination_via_fm_legend_surface(cat_df, orient, direction):
    """The ``fm.Legend(direction=...)`` twin of the theme-route test above.

    Python-half regression (F-L04-04): ``fm.Legend(direction="vertical")``
    used to serialize to ``{}`` because ``_LEGEND_DEFAULTS`` stripped an
    explicit value that textually matched the Python default, so this
    surface never reached two of the eight orient×direction combinations
    (``direction="vertical"`` paired with any orient) even after D5's Rust
    fix landed. The theme route (``fm.Theme(legend_direction=...)``) already
    reached all eight and is covered above; this proves the per-channel
    ``fm.Legend`` surface now does too.
    """
    chart = (
        fm.Chart(cat_df)
        .mark_point()
        .encode(
            x="x:Q",
            y="y:Q",
            color=fm.Color("cat:N", legend=fm.Legend(orient=orient, direction=direction)),
        )
        .properties(**WIDE)
    )
    svg, msgs = render(chart)
    drawn = texts(svg)
    for label in CATEGORIES:
        assert label in drawn, (
            f"orient={orient} direction={direction}: legend entry {label!r} was dropped; "
            f"drawn text nodes = {drawn}"
        )
    assert not [m for m in msgs if "legend overflowed" in m], (
        f"orient={orient} direction={direction} overflowed at {WIDE}: {msgs}"
    )


@pytest.mark.parametrize("orient", ["right", "left"])
def test_horizontal_direction_on_a_side_legend_lays_entries_in_a_row(cat_df, orient):
    """``direction`` genuinely re-arranges: on a side legend the entries move
    from a column to a row (distinct x, shared y) — not merely "all present"."""
    chart = (
        fm.Chart(cat_df)
        .mark_point()
        .encode(x="x:Q", y="y:Q", color="cat:N")
        .theme(fm.Theme(legend_direction="horizontal"))
        .configure_legend(orient=orient)
        .properties(**WIDE)
    )
    svg, _ = render(chart)
    anchors = [(x, y) for x, y, t in _label_anchors_xy(svg) if t in CATEGORIES]
    ys = {y for _, y in anchors}
    xs = {x for x, _ in anchors}
    assert len(ys) == 1, f"horizontal entries must share one baseline; got {sorted(ys)}"
    assert len(xs) == len(CATEGORIES), f"horizontal entries need distinct x; got {sorted(xs)}"


@pytest.mark.parametrize("orient", ["top", "bottom"])
def test_vertical_direction_on_a_strip_legend_stacks_entries(cat_df, orient):
    """The mirror case: a top/bottom strip with ``direction="vertical"``
    stacks its entries (distinct y, shared x)."""
    chart = (
        fm.Chart(cat_df)
        .mark_point()
        .encode(x="x:Q", y="y:Q", color="cat:N")
        .theme(fm.Theme(legend_direction="vertical"))
        .configure_legend(orient=orient)
        .properties(**WIDE)
    )
    svg, _ = render(chart)
    anchors = [(x, y) for x, y, t in _label_anchors_xy(svg) if t in CATEGORIES]
    ys = {y for _, y in anchors}
    xs = {x for x, _ in anchors}
    assert len(ys) == len(CATEGORIES), f"vertical entries need distinct y; got {sorted(ys)}"
    assert len(xs) == 1, f"vertical entries must share one x; got {sorted(xs)}"


def _label_anchors_xy(svg: str) -> list[tuple[float, float, str]]:
    """``(x, y, content)`` for every ``<text>`` node that carries both anchors.

    The entry-arrangement assertions need the anchor GEOMETRY, not just the
    text: "every label present" is satisfied by a legend that stacks entries
    the wrong way, so the direction tests read the anchors instead.
    """
    return [
        (float(m.group(1)), float(m.group(2)), m.group(3))
        for m in re.finditer(r'<text x="([-\d.]+)" y="([-\d.]+)"[^>]*>([^<]+)</text>', svg)
    ]


# ---------------------------------------------------------------------------
# D5 — horizontal colorbar
# ---------------------------------------------------------------------------


def test_horizontal_colorbar_runs_left_to_right_with_ticks_beneath(num_df):
    """``direction="horizontal"`` on a continuous color scale renders a
    horizontal gradient bar. Pinned on three independent signals so a partial
    implementation (e.g. a transposed rect but a still-vertical gradient)
    fails: the gradient vector, the bar aspect, and the tick placement."""
    chart = (
        fm.Chart(num_df)
        .mark_point()
        .encode(x="x:Q", y="y:Q", color="val:Q")
        .theme(fm.Theme(legend_direction="horizontal"))
        .properties(**WIDE)
    )
    svg, _ = render(chart)
    grad = re.search(
        r'<linearGradient id="ferrum-colorbar-0" x1="(\d+)" y1="(\d+)" x2="(\d+)" y2="(\d+)"',
        svg,
    )
    assert grad, "expected a colorbar gradient definition"
    assert grad.groups() == ("0", "0", "1", "0"), (
        f"horizontal colorbar gradient must run left->right, got {grad.groups()}"
    )
    bar = re.search(
        r'<rect x="([-\d.]+)" y="([-\d.]+)" width="([-\d.]+)" height="([-\d.]+)" '
        r'fill="url\(#ferrum-colorbar-0\)"',
        svg,
    )
    assert bar, "expected the gradient-filled colorbar rect"
    bar_x, bar_y, bar_w, bar_h = (float(g) for g in bar.groups())
    assert bar_w > bar_h, f"horizontal bar must be wider than thick; got {bar_w}x{bar_h}"
    ticks = [(x, y, t) for x, y, t in _label_anchors_xy(svg) if t in {"0", "25", "50", "75", "100"}]
    assert len(ticks) >= 2, f"expected colorbar tick labels; got {texts(svg)}"
    assert len({x for x, _, _ in ticks}) == len(ticks), "tick labels must spread along x"
    assert all(y > bar_y + bar_h for _, y, _ in ticks), (
        "horizontal colorbar tick labels must sit beneath the bar"
    )
    assert 'text-anchor="middle"' in svg, "horizontal tick labels are centered on their tick"


def test_vertical_colorbar_is_unchanged_by_the_horizontal_work(num_df):
    """The default (vertical) colorbar keeps its bottom→top gradient and
    right-hand tick labels — the byte-identity half of the D5 colorbar change."""
    chart = fm.Chart(num_df).mark_point().encode(x="x:Q", y="y:Q", color="val:Q").properties(**WIDE)
    svg, _ = render(chart)
    grad = re.search(
        r'<linearGradient id="ferrum-colorbar-0" x1="(\d+)" y1="(\d+)" x2="(\d+)" y2="(\d+)"',
        svg,
    )
    assert grad and grad.groups() == ("0", "1", "0", "0")
    bar = re.search(
        r'<rect x="([-\d.]+)" y="([-\d.]+)" width="([-\d.]+)" height="([-\d.]+)" '
        r'fill="url\(#ferrum-colorbar-0\)"',
        svg,
    )
    assert bar
    _, _, bar_w, bar_h = (float(g) for g in bar.groups())
    assert bar_h > bar_w, f"vertical bar must be taller than wide; got {bar_w}x{bar_h}"


def test_colorbar_direction_defaults_from_orient_with_no_explicit_direction(num_df):
    """Disclosed behavior change: a colorbar's DEFAULT direction now follows
    ``orient`` the same way the categorical arm's always has, when the caller
    sets no ``direction`` at all (not even via ``fm.Theme``).

    Before this batch a colorbar was unconditionally drawn vertical
    regardless of ``orient`` — only an *explicit* ``direction="horizontal"``
    (pinned above) produced a horizontal bar. This is the render-level pin
    for that default itself: ``orient="bottom"`` alone must now be enough,
    and ``orient="right"`` (the default) must still be enough to keep it
    vertical. The two assertions are geometrically exclusive (a bar cannot
    be both wider-than-tall and taller-than-wide), so this test is RED
    against the disclosed judgment call's stated one-line revert (pinning
    ``layout_colorbar`` to always resolve ``Vertical``) without needing to
    touch ``crates/`` to prove it: that revert leaves every OTHER test in
    this file green (none of them omit ``direction`` on a bottom/top
    colorbar) but flips ``bar_w > bar_h`` to false right here.
    """

    def bar_dims(chart) -> tuple[float, float]:
        svg, _ = render(chart)
        bar = re.search(
            r'<rect x="([-\d.]+)" y="([-\d.]+)" width="([-\d.]+)" height="([-\d.]+)" '
            r'fill="url\(#ferrum-colorbar-0\)"',
            svg,
        )
        assert bar, f"expected the gradient-filled colorbar rect; svg={svg[:500]}"
        _, _, w, h = (float(g) for g in bar.groups())
        return w, h

    bottom_w, bottom_h = bar_dims(
        fm.Chart(num_df)
        .mark_point()
        .encode(x="x:Q", y="y:Q", color="val:Q")
        .configure_legend(orient="bottom")
        .properties(**WIDE)
    )
    assert bottom_w > bottom_h, (
        f"orient='bottom' with no explicit direction must default to a horizontal "
        f"colorbar; got {bottom_w}x{bottom_h}"
    )

    right_w, right_h = bar_dims(
        fm.Chart(num_df)
        .mark_point()
        .encode(x="x:Q", y="y:Q", color="val:Q")
        .configure_legend(orient="right")
        .properties(**WIDE)
    )
    assert right_h > right_w, (
        f"orient='right' with no explicit direction must stay a vertical colorbar; "
        f"got {right_w}x{right_h}"
    )


# ---------------------------------------------------------------------------
# D6 / F-L04-05 — categorical `values`
# ---------------------------------------------------------------------------


def test_values_filters_categorical_legend_entries(cat_df):
    """``Legend(values=[...])`` keeps only the named categories.

    F-L04-05's repro: the two charts below were byte-identical before D6.
    """

    def chart(values):
        return (
            fm.Chart(cat_df)
            .mark_point()
            .encode(x="x:Q", y="y:Q", color=fm.Color("cat:N", legend=fm.Legend(values=values)))
            .properties(**WIDE)
        )

    one, _ = render(chart(["alpha"]))
    all_three, _ = render(chart(CATEGORIES))
    assert one != all_three, "values= must change the rendered legend"
    drawn = texts(one)
    assert "alpha" in drawn
    assert "beta" not in drawn and "gamma" not in drawn, drawn


def test_values_orders_categorical_legend_entries(cat_df):
    """``values`` orders as well as filters — the entries come out in the
    order the caller wrote them, not the scale-domain order."""
    chart = (
        fm.Chart(cat_df)
        .mark_point()
        .encode(
            x="x:Q",
            y="y:Q",
            color=fm.Color("cat:N", legend=fm.Legend(values=["gamma", "alpha"])),
        )
        .properties(**WIDE)
    )
    svg, _ = render(chart)
    drawn = [t for t in texts(svg) if t in CATEGORIES]
    assert drawn == ["gamma", "alpha"], drawn


def test_values_naming_an_unknown_category_warns_and_skips(cat_df):
    """A `values` entry matching no category has no swatch to draw. Per the
    batch's sanctioned-degradation rule it is skipped with a stable warning,
    never silently ignored and never invented as an empty swatch."""
    chart = (
        fm.Chart(cat_df)
        .mark_point()
        .encode(
            x="x:Q",
            y="y:Q",
            color=fm.Color("cat:N", legend=fm.Legend(values=["alpha", "delta"])),
        )
        .properties(**WIDE)
    )
    svg, msgs = render(chart)
    assert any("delta" in m and "match no legend entry" in m for m in msgs), msgs
    drawn = [t for t in texts(svg) if t in CATEGORIES or t == "delta"]
    assert drawn == ["alpha"], drawn


def test_values_absent_leaves_the_categorical_legend_untouched(cat_df):
    """The byte-identity half of D6: no ``values`` → no change at all."""
    with_empty_legend = (
        fm.Chart(cat_df)
        .mark_point()
        .encode(x="x:Q", y="y:Q", color=fm.Color("cat:N", legend=fm.Legend()))
        .properties(**WIDE)
    )
    plain = fm.Chart(cat_df).mark_point().encode(x="x:Q", y="y:Q", color="cat:N").properties(**WIDE)
    assert render(with_empty_legend)[0] == render(plain)[0]


def test_values_on_a_colorbar_still_replaces_tick_labels(num_df):
    """D6 must not disturb the colorbar arm's pre-existing ``values``
    semantics (explicit tick labels)."""
    chart = (
        fm.Chart(num_df)
        .mark_point()
        .encode(
            x="x:Q",
            y="y:Q",
            color=fm.Color("val:Q", legend=fm.Legend(values=["lo", "hi"])),
        )
        .properties(**WIDE)
    )
    svg, msgs = render(chart)
    drawn = texts(svg)
    assert "lo" in drawn and "hi" in drawn, drawn
    assert not [m for m in msgs if "match no legend entry" in m], (
        f"a colorbar has no entries to match against; must not warn: {msgs}"
    )


# ---------------------------------------------------------------------------
# NF-B13 — X/Y(legend=...) has a real consumer
# ---------------------------------------------------------------------------


def test_x_channel_legend_orient_moves_the_legend(cat_df):
    """``X(legend=Legend(orient=...))`` reaches the same per-channel override
    path ``Color(legend=...)`` uses. Before NF-B13 the kwarg serialized to the
    wire and nothing read it, so these two rendered byte-identically."""

    def chart(x_channel):
        return (
            fm.Chart(cat_df)
            .mark_point()
            .encode(x=x_channel, y="y:Q", color="cat:N")
            .properties(**WIDE)
        )

    moved, _ = render(chart(fm.X("x", type_="Q", legend=fm.Legend(orient="bottom"))))
    plain, _ = render(chart(fm.X("x", type_="Q")))
    assert moved != plain, "X(legend=orient=bottom) must change the render"
    # The legend labels move below the plot area rather than beside it.
    moved_ys = [y for _, y, t in _label_anchors_xy(moved) if t in CATEGORIES]
    plain_ys = [y for _, y, t in _label_anchors_xy(plain) if t in CATEGORIES]
    assert min(moved_ys) > max(plain_ys), (
        f"bottom-oriented legend must sit lower; moved={moved_ys} plain={plain_ys}"
    )


def test_y_channel_legend_title_is_honored(cat_df):
    """The cascade covers ``Y`` too, and every field — not just ``orient``."""
    chart = (
        fm.Chart(cat_df)
        .mark_point()
        .encode(x="x:Q", y=fm.Y("y", type_="Q", legend=fm.Legend(title="from y")), color="cat:N")
        .properties(**WIDE)
    )
    svg, _ = render(chart)
    assert "from y" in texts(svg), texts(svg)


def test_color_channel_legend_beats_a_positional_one(cat_df):
    """Precedence within the per-channel level is color > x > y: the color
    channel owns the legend surface, so its own value wins."""
    chart = (
        fm.Chart(cat_df)
        .mark_point()
        .encode(
            x=fm.X("x", type_="Q", legend=fm.Legend(title="from x")),
            y="y:Q",
            color=fm.Color("cat:N", legend=fm.Legend(title="from color")),
        )
        .properties(**WIDE)
    )
    svg, _ = render(chart)
    drawn = texts(svg)
    assert "from color" in drawn, drawn
    assert "from x" not in drawn, drawn


def test_x_legend_none_suppresses_the_chart_legend(cat_df):
    """``X(legend=None)`` suppresses the whole chart's legend.

    Disclosed behavior change (NF-B13): before X/Y gained a real legend
    consumer, ``disabled`` was the one field of the positional ``legend=``
    dict nothing read, so ``X(legend=None)`` (which normalizes to
    ``{"disabled": True}``) did nothing. Routing every field of the
    per-channel legend override through one cascade — the alternative of
    special-casing ``disabled`` out — would leave positional.py's documented
    "``None``/``False`` to suppress" claim false, which this batch forbids.
    """
    chart = (
        fm.Chart(cat_df)
        .mark_point()
        .encode(x=fm.X("x", type_="Q", legend=None), y="y:Q", color="cat:N")
        .properties(**WIDE)
    )
    svg, _ = render(chart)
    drawn = texts(svg)
    for label in CATEGORIES:
        assert label not in drawn, f"X(legend=None) must suppress the legend; got {drawn}"


def test_y_legend_false_suppresses_the_chart_legend(cat_df):
    """The ``Y`` twin, and the ``False`` spelling rather than ``None``."""
    chart = (
        fm.Chart(cat_df)
        .mark_point()
        .encode(x="x:Q", y=fm.Y("y", type_="Q", legend=False), color="cat:N")
        .properties(**WIDE)
    )
    svg, _ = render(chart)
    drawn = texts(svg)
    for label in CATEGORIES:
        assert label not in drawn, f"Y(legend=False) must suppress the legend; got {drawn}"


def test_x_legend_dict_override_takes_effect(cat_df):
    """``X(legend={...})`` (a raw dict, not an ``fm.Legend`` instance) reaches
    the same consumer — the per-channel override path is not
    ``fm.Legend``-only."""
    chart = (
        fm.Chart(cat_df)
        .mark_point()
        .encode(x=fm.X("x", type_="Q", legend={"title": "from x dict"}), y="y:Q", color="cat:N")
        .properties(**WIDE)
    )
    svg, _ = render(chart)
    assert "from x dict" in texts(svg), texts(svg)


# ---------------------------------------------------------------------------
# D5 — per-channel orient="none"
# ---------------------------------------------------------------------------


def test_per_channel_orient_none_suppresses_the_legend(cat_df):
    """``fm.Legend(orient="none")`` disables that channel's legend, matching
    the chart-level spelling (which Python already resolves to ``disabled``)."""

    def chart(legend):
        return (
            fm.Chart(cat_df)
            .mark_point()
            .encode(x="x:Q", y="y:Q", color=fm.Color("cat:N", legend=legend))
            .properties(**WIDE)
        )

    suppressed, _ = render(chart(fm.Legend(orient="none")))
    disabled, _ = render(chart(None))
    drawn = texts(suppressed)
    for label in CATEGORIES:
        assert label not in drawn, f"orient='none' must suppress the legend; got {drawn}"
    assert suppressed == disabled, "orient='none' and legend=None are one suppression"


def test_higher_precedence_placement_beats_lower_precedence_orient_none(cat_df):
    """Suppression obeys the same color > x > y precedence every other legend
    field does: a lower-precedence channel cannot blank a legend a
    higher-precedence one explicitly placed.

    RED before the cycle-2 fix, which read the chain with "any channel
    suppresses" while reading every other field first-Some — so the same
    ``orient`` field answered at two different precedences inside one function
    and the legend vanished entirely.
    """
    conflict = (
        fm.Chart(cat_df)
        .mark_point()
        .encode(
            x="x:Q",
            y=fm.Y("y", type_="Q", legend=fm.Legend(orient="none")),
            color=fm.Color("cat:N", legend=fm.Legend(orient="right")),
        )
        .properties(**WIDE)
    )
    svg, _ = render(conflict)
    drawn = texts(svg)
    for label in CATEGORIES:
        assert label in drawn, (
            f"color's explicit orient='right' must outrank y's orient='none'; got {drawn}"
        )


def test_orient_none_still_wins_when_no_higher_channel_places_the_legend(cat_df):
    """The mirror of the pin above — the fix must not make ``orient="none"``
    on a positional channel inert. With ``color`` expressing no ``orient``
    opinion, ``y``'s suppression is the first ``Some`` and still wins."""
    chart = (
        fm.Chart(cat_df)
        .mark_point()
        .encode(
            x="x:Q",
            y=fm.Y("y", type_="Q", legend=fm.Legend(orient="none")),
            color=fm.Color("cat:N", legend=fm.Legend(title="t")),
        )
        .properties(**WIDE)
    )
    svg, _ = render(chart)
    drawn = texts(svg)
    for label in CATEGORIES:
        assert label not in drawn, f"y's orient='none' must still suppress; got {drawn}"


# ---------------------------------------------------------------------------
# D7 — cascade repair
# ---------------------------------------------------------------------------


def test_per_channel_orient_beats_configure_legend_orient(cat_df):
    """Disclosed behavior change: per-channel now wins on ``orient``.

    Before D7, ``apply_chart_config`` ran after the per-channel write and
    clobbered it, so the chart below rendered its legend on the right.
    """
    conflict = (
        fm.Chart(cat_df)
        .mark_point()
        .encode(x="x:Q", y="y:Q", color=fm.Color("cat:N", legend=fm.Legend(orient="bottom")))
        .configure_legend(orient="right")
        .properties(**WIDE)
    )
    per_channel_only = (
        fm.Chart(cat_df)
        .mark_point()
        .encode(x="x:Q", y="y:Q", color=fm.Color("cat:N", legend=fm.Legend(orient="bottom")))
        .properties(**WIDE)
    )
    assert render(conflict)[0] == render(per_channel_only)[0], (
        "a conflicting chart-level orient must not override the per-channel one"
    )


def test_per_channel_columns_and_title_font_size_beat_chart_level(cat_df):
    """The other two inverted fields, pinned the same way."""
    conflict = (
        fm.Chart(cat_df)
        .mark_point()
        .encode(
            x="x:Q",
            y="y:Q",
            color=fm.Color("cat:N", legend=fm.Legend(columns=3, title_font_size=20)),
        )
        .configure_legend(columns=1, title_font_size=9)
        .properties(**WIDE)
    )
    per_channel_only = (
        fm.Chart(cat_df)
        .mark_point()
        .encode(
            x="x:Q",
            y="y:Q",
            color=fm.Color("cat:N", legend=fm.Legend(columns=3, title_font_size=20)),
        )
        .properties(**WIDE)
    )
    assert render(conflict)[0] == render(per_channel_only)[0]


def test_chart_level_legend_fields_still_apply_when_no_per_channel_value(cat_df):
    """The cascade is fill-only, not per-channel-only: chart level still works."""
    base = fm.Chart(cat_df).mark_point().encode(x="x:Q", y="y:Q", color="cat:N")
    plain, _ = render(base.properties(**WIDE))
    bottom, _ = render(base.configure_legend(orient="bottom").properties(**WIDE))
    assert plain != bottom, "configure_legend(orient=) must still take effect"


def test_configure_legend_label_font_size_does_not_resize_axis_labels(cat_df):
    """D7: ``configure_legend(label_font_size=)`` gets a legend-own slot.

    It used to write ``theme.typography.label_font_size``, the SHARED slot axis
    tick labels read — so a legend knob silently resized the axes. Pinned on
    the axis tick labels' font-size attribute specifically, which is the thing
    that must NOT move, plus the legend labels', which must.
    """
    base = fm.Chart(cat_df).mark_point().encode(x="x:Q", y="y:Q", color="cat:N")
    plain, _ = render(base.properties(**WIDE))
    resized, _ = render(base.configure_legend(label_font_size=22).properties(**WIDE))

    def font_size_of(svg: str, label: str) -> float:
        m = re.search(rf'<text[^>]*font-size="([\d.]+)"[^>]*>{label}</text>', svg)
        assert m, f"no <text> node for {label!r}"
        return float(m.group(1))

    assert font_size_of(resized, "alpha") == 22.0, "legend entry labels must resize"
    # "1" is an x/y axis tick label in this fixture's domain.
    assert font_size_of(resized, "1") == font_size_of(plain, "1"), (
        "a legend font knob must not resize axis tick labels"
    )


def test_configure_legend_label_font_size_resizes_aux_legend_labels(num_df):
    """The aux (size/shape/stroke-dash) half of the same knob.

    Moving ``configure_legend(label_font_size=)`` off the shared typography
    slot left aux legends RESERVING space at the override size while still
    PAINTING at the theme size — an over-wide gutter with small text. Pinned on
    both halves together, because either alone passes on the broken build: the
    gutter assertion held before the fix and the font assertion is what failed.
    """
    base = fm.Chart(num_df).mark_point().encode(x="x:Q", y="y:Q", size="val:Q")
    plain, _ = render(base.properties(**WIDE))
    resized, _ = render(base.configure_legend(label_font_size=22).properties(**WIDE))

    def aux_label_anchors(svg: str) -> list[tuple[float, float]]:
        """``(x, font_size)`` for the size-legend entry labels.

        Identified as the numeric text nodes sharing the single rightmost x
        anchor: the aux block is a vertical stack in the right gutter, so all
        its entry labels are left-aligned on one x, while axis tick labels sit
        further left and each at its own x. Filtering on a coarse "right of
        75% of the canvas" band instead would sweep in the last x-axis tick.
        """
        found = [
            (float(m.group(1)), float(m.group(2)))
            for m in re.finditer(
                r'<text x="([-\d.]+)" y="[-\d.]+" fill="[^"]*" font-family="[^"]*" '
                r'font-size="([\d.]+)"[^>]*>[\d.]+</text>',
                svg,
            )
        ]
        assert found, "expected numeric text nodes"
        rightmost = max(x for x, _ in found)
        return [(x, fs) for x, fs in found if x == rightmost]

    plain_aux = aux_label_anchors(plain)
    resized_aux = aux_label_anchors(resized)
    assert plain_aux, "expected size-legend entry labels in the right gutter"
    assert resized_aux, "expected size-legend entry labels in the right gutter"
    assert all(fs == 11.0 for _, fs in plain_aux), plain_aux
    assert all(fs == 22.0 for _, fs in resized_aux), (
        f"aux legend labels must paint at the override size; got {resized_aux}"
    )
    # And the gutter it reserved genuinely widened (labels start further left),
    # so the reservation and the painted size describe the same block.
    assert min(x for x, _ in resized_aux) < min(x for x, _ in plain_aux), (
        "a bigger aux label font must reserve a wider gutter"
    )


def test_configure_axis_label_font_size_still_resizes_axis_labels(cat_df):
    """The half D7 must not break: the axis knob still owns axis label size."""
    base = fm.Chart(cat_df).mark_point().encode(x="x:Q", y="y:Q", color="cat:N")
    plain, _ = render(base.properties(**WIDE))
    resized, _ = render(base.configure_axis(label_font_size=22).properties(**WIDE))
    assert plain != resized
    assert re.search(r'<text[^>]*font-size="22"[^>]*>1</text>', resized), (
        "configure_axis(label_font_size=) must still size axis tick labels"
    )


# ---------------------------------------------------------------------------
# Byte-identity for untouched surfaces
# ---------------------------------------------------------------------------


def test_default_legend_chart_is_unaffected_by_the_legend_work(cat_df):
    """A chart naming none of the touched fields renders the same as one that
    names them redundantly (each set to the value already in force)."""
    plain = fm.Chart(cat_df).mark_point().encode(x="x:Q", y="y:Q", color="cat:N").properties(**WIDE)
    redundant = (
        fm.Chart(cat_df)
        .mark_point()
        .encode(x="x:Q", y="y:Q", color="cat:N")
        .configure_legend(orient="right")
        .properties(**WIDE)
    )
    assert render(plain)[0] == render(redundant)[0]


def test_a_chart_with_no_legend_at_all_is_unaffected(cat_df):
    """The no-legend path (no color encoding) must not have moved."""
    svg, msgs = render(fm.Chart(cat_df).mark_point().encode(x="x:Q", y="y:Q").properties(**WIDE))
    assert "<svg" in svg
    assert not [m for m in msgs if "legend" in m], msgs


# ---------------------------------------------------------------------------
# Token validation — one shared orient/direction vocabulary, loud on every surface
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "field, valid, invalid",
    [("orient", "bottom", "diagonal"), ("direction", "horizontal", "sideways")],
)
def test_fm_legend_refuses_an_unrecognized_token(field, valid, invalid):
    """``fm.Legend`` refuses immediately at construction, not silently at render."""
    fm.Legend(**{field: valid})  # the valid token does not raise
    with pytest.raises(ValueError, match=field):
        fm.Legend(**{field: invalid})


@pytest.mark.parametrize(
    "field, valid, invalid",
    [("orient", "bottom", "diagonal"), ("direction", "horizontal", "sideways")],
)
def test_configure_legend_refuses_an_unrecognized_token(field, valid, invalid):
    """The chart-level ``configure_legend``/``LegendConfig`` surface, same vocabulary.

    ``orient`` already validated before this batch (``LegendConfig.orient``);
    ``direction`` did not — it silently reached Rust's total parser and fell
    back to the theme default on a typo. Both now refuse the same way.
    """
    fm.Chart(pl.DataFrame({"x": [1]})).mark_point().encode(x="x").configure_legend(
        **{field: valid}
    )  # the valid token does not raise
    with pytest.raises(ValueError, match=field):
        fm.Chart(pl.DataFrame({"x": [1]})).mark_point().encode(x="x").configure_legend(
            **{field: invalid}
        )


@pytest.mark.parametrize(
    "field, valid, invalid",
    [("orient", "bottom", "diagonal"), ("direction", "horizontal", "sideways")],
)
def test_raw_legend_dict_refuses_an_unrecognized_token(cat_df, field, valid, invalid):
    """A raw ``legend={...}`` dict (not an ``fm.Legend`` instance) gets the
    same loud check — the validator lives at the one dict-normalize
    chokepoint every channel's ``legend=`` kwarg routes through.

    Unlike ``fm.Legend``/``configure_legend`` (which validate at
    construction), a raw dict is only normalized when the encoding spec is
    built — ``to_svg()`` — so this pin exercises the render entry point
    rather than ``.encode()`` itself.
    """

    def chart(value):
        return (
            fm.Chart(cat_df)
            .mark_point()
            .encode(x="x:Q", y="y:Q", color=fm.Color("cat:N", legend={field: value}))
            .to_svg()
        )

    chart(valid)  # the valid token does not raise
    with pytest.raises(ValueError, match=field):
        chart(invalid)
