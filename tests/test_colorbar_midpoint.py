"""Diverging colorbar midpoint and scheme-selection acceptance tests.

A diverging colorbar shows the same color at a value as the cells it colors:
``heatmap(center=0)`` places the scheme's mid color at the midpoint's position
on the bar, so every annotated cell matches the bar within a small per-channel
tolerance; ``heatmap(center=0.5)`` with no ``cmap`` picks the theme's
diverging scheme; and ``configure_color(domain=, range=)`` updates the bar
along with the marks it colors. Everything with no midpoint stays
byte-identical to before (proven at handback by SVG hashes against the stage
baseline, not read from a committed test).

Every comparison here reads the bar's stops and the marks' own rendered fills
from the SAME render -- never a value independently recomputed from the
scheme/midpoint math, which belongs to Rust (`render/prepare/legend.rs`,
`render/scale_resolve/color.rs`) and must not be duplicated in Python. This
also discriminates a domain-symmetrizing implementation of the midpoint: such
an implementation would make two charts with the same domain but different
``domain_mid`` render identically, which the tests below assert they do not.
"""

from __future__ import annotations

import re

import numpy as np
import polars as pl
import pytest

import ferrum as fm
from tests._svg_rects import figure_elements, plot_rect, svg_elements

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_STOP_RE = re.compile(r'<stop offset="([\d.]+)" stop-color="(#[0-9a-fA-F]+)"')


def _gradient_stops(svg: str) -> list[tuple[float, str]]:
    """Every `<stop offset=... stop-color=...>` in *svg*, sorted by offset."""
    return sorted((float(o), hexstr) for o, hexstr in _STOP_RE.findall(svg))


def _rgb(hexstr: str) -> tuple[int, int, int]:
    h = hexstr.lstrip("#")
    return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))


def _within(a: tuple[int, int, int], b: tuple[int, int, int], tol: int) -> bool:
    return all(abs(x - y) <= tol for x, y in zip(a, b))


def _bar_color_at(
    stops: list[tuple[float, str]], lo: float, hi: float, v: float
) -> tuple[int, int, int]:
    """The color an SVG renderer paints at value *v*'s linear position on the
    bar: standard `linearGradient` semantics, linear RGB interpolation between
    the two stops bracketing that position. This reproduces the SVG spec's own
    rendering rule, not ferrum's scheme/midpoint math."""
    if hi <= lo:
        p = 0.5
    else:
        p = min(max((v - lo) / (hi - lo), stops[0][0]), stops[-1][0])
    for (o0, h0), (o1, h1) in zip(stops, stops[1:]):
        if o0 <= p <= o1:
            t = 0.0 if o1 == o0 else (p - o0) / (o1 - o0)
            c0, c1 = _rgb(h0), _rgb(h1)
            r = round(c0[0] + t * (c1[0] - c0[0]))
            g = round(c0[1] + t * (c1[1] - c0[1]))
            b = round(c0[2] + t * (c1[2] - c0[2]))
            return (r, g, b)
    return _rgb(stops[-1][1])


def _point_fills(svg: str) -> list[str]:
    """`fill` of every rendered `<circle>`, in document order."""
    return [a["fill"] for a in svg_elements(svg, "circle", lambda a: "fill" in a)]


def _legend_tick_labels(svg: str) -> list[str]:
    """The colorbar's own tick labels, excluding the legend title text that
    precedes them: every `text-anchor="start"` text is either the title or a
    tick, and the title is always first."""
    labels = re.findall(r'<text[^>]*text-anchor="start"[^>]*>([^<]*)</text>', svg)
    return labels[1:]


def _assert_bar_label_endpoints(svg: str, lo: float, hi: float, *, tol: float = 0.1) -> None:
    """The colorbar's own tick-label endpoints equal `[lo, hi]`. A bar whose
    stops/swatches follow a `configure_color(domain=)` override but whose
    LABELS still show the stale pre-config domain would pass a stops-only or
    marks-only check; this is the one assertion that catches that."""
    ticks = [float(x) for x in _legend_tick_labels(svg)]
    assert ticks[0] == pytest.approx(lo, abs=tol), ticks
    assert ticks[-1] == pytest.approx(hi, abs=tol), ticks


def _is_number(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False


def _annotated_cells(svg: str) -> list[tuple[float, tuple[int, int, int]]]:
    """`(value, rendered_fill)` for every `heatmap(annot=True)` cell, paired by
    position: each numeric text inside the plot area is a cell's own
    annotation, matched to the one cell `<rect>` it sits inside. Independent of
    draw order, so it does not assume anything about row/column emission
    order."""
    els = figure_elements(svg)
    rects = [
        e
        for e in els
        if e["tag"] == "rect"
        and e["panel"] is not None
        and e.get("fill")
        and not e["fill"].startswith("url(")
    ]
    px, py, pw, ph = plot_rect(svg)
    texts = [
        e
        for e in els
        if e["tag"] == "text"
        and _is_number(e["text"])
        and px <= e["x"] <= px + pw
        and py <= e["y"] <= py + ph
    ]
    out = []
    for t in texts:
        matches = [
            r
            for r in rects
            if r["x"] <= t["x"] <= r["x"] + r["w"] and r["y"] <= t["y"] <= r["y"] + r["h"]
        ]
        assert len(matches) == 1, f"expected exactly one cell under annotation {t!r}, got {matches}"
        out.append((float(t["text"]), _rgb(matches[0]["fill"])))
    return out


def _docs_correlation_matrix() -> np.ndarray:
    """Reconstructs ``design-docs/demos/demo_diagnostics.py``'s ``df_corr``
    correlation matrix bit-for-bit, by replaying the same ``np.random.default_rng(42)``
    draw sequence (box/swarm/bar/hist draws consume rng state before the corr
    matrix's own draw). The "within 8 per channel" bound is not a general
    property of 11-stop gradient sampling -- it is sensitive to exactly where
    a cell's value falls relative to the scheme's steepest region, so the
    acceptance check below is run against the data the decision's own bound
    was measured on, not a differently-seeded substitute whose values could
    land anywhere on the curve."""
    rng = np.random.default_rng(42)
    for _group, mu in zip(
        ["Control", "Treatment A", "Treatment B", "Treatment C"], [0.0, 0.8, -0.4, 1.2]
    ):
        for _ in range(80):
            rng.normal(mu, 1.0)
    for mu, sigma, n in [(0, 1, 40), (1, 0.8, 40), (-0.5, 1.2, 40)]:
        rng.normal(mu, sigma, n)
    for _ in range(4):
        for _ in range(3):
            rng.integers(40, 120)
    for g in ["Group A", "Group B"]:
        for _ in range(150):
            rng.normal(0.0 if g == "Group A" else 2.0, 1.0)
    cov_raw = rng.standard_normal((5, 50))
    return np.corrcoef(cov_raw)


def _docs_correlation_df() -> tuple[pl.DataFrame, float, float]:
    corr = np.round(_docs_correlation_matrix(), 2)
    features = ["Feature A", "Feature B", "Feature C", "Feature D", "Feature E"]
    rows = [
        {"feature": features[i], **{features[j]: float(corr[i, j]) for j in range(len(features))}}
        for i in range(len(features))
    ]
    return pl.DataFrame(rows), float(corr.min()), float(corr.max())


# ---------------------------------------------------------------------------
# heatmap(center=0) on the spec's/docs heatmap data
# ---------------------------------------------------------------------------


def test_heatmap_center_zero_every_annotated_cell_matches_bar_within_tolerance():
    df, lo, hi = _docs_correlation_df()
    svg = fm.heatmap(df, center=0.0, annot=True, fmt=".2f").to_svg()
    stops = _gradient_stops(svg)
    cells = _annotated_cells(svg)
    assert len(cells) == 25, f"expected 25 annotated cells (5x5), got {len(cells)}"
    for value, fill in cells:
        bar = _bar_color_at(stops, lo, hi, value)
        assert _within(fill, bar, tol=8), f"value {value}: cell {fill} vs bar {bar}; stops {stops}"


def test_heatmap_center_zero_mid_stop_equals_the_schemes_mid_color():
    """The p_mid stop equals the scheme's mid color (``scheme.sample(0.5)``):
    proven by comparing it to the 0.5-offset stop of a NO-midpoint render of
    the SAME scheme, not by comparing two midpoint renders to each other --
    which would also pass if both sampled the scheme at p_mid instead of
    0.5, a different bug neither render would discriminate against the
    other."""
    df_a = pl.DataFrame(
        {"a": [-1.0, 10.0], "b": [0.0, 0.0]}
    )  # domain [-1, 10], mid 3 -> p_mid = 4/11
    svg_a = fm.heatmap(df_a, annot=False, cmap="rdbu", vmin=-1.0, vmax=10.0, center=3.0).to_svg()
    p_mid = round((3.0 - -1.0) / (10.0 - -1.0), 4)
    stop_a = dict(_gradient_stops(svg_a))
    assert p_mid in stop_a
    no_midpoint_svg = _mid_chart(0.0, 1.0, None, scheme="rdbu").to_svg()
    true_mid = dict(_gradient_stops(no_midpoint_svg))[0.5]
    assert stop_a[p_mid] == true_mid


# ---------------------------------------------------------------------------
# DivergingScale(domain_mid=) on a non-heatmap mark
# ---------------------------------------------------------------------------


def _mid_chart(lo: float, hi: float, mid: float | None, *, scheme: str | None = None) -> fm.Chart:
    df = pl.DataFrame(
        {
            "x": [1.0, 2.0, 3.0],
            "y": [1.0, 2.0, 3.0],
            "v": [lo, mid if mid is not None else (lo + hi) / 2, hi],
        }
    )
    return (
        fm.Chart(df)
        .mark_point()
        .encode(
            x="x:Q",
            y="y:Q",
            color=fm.Color(
                "v", scale=fm.DivergingScale(scheme=scheme, domain=[lo, hi], domain_mid=mid)
            ),
        )
        .properties(width=300, height=200)
    )


def test_diverging_scale_domain_mid_on_point_mark_matches_the_bar_at_the_midpoint():
    """A point drawn exactly at the midpoint value is colored scheme.sample(0.5),
    the same color the colorbar's inserted p_mid stop carries -- checked on a
    plain `mark_point` chart, not a heatmap."""
    lo, mid, hi = -1.0, 3.0, 10.0
    svg = _mid_chart(lo, hi, mid).to_svg()
    stops = dict(_gradient_stops(svg))
    p_mid = round((mid - lo) / (hi - lo), 4)
    assert p_mid in stops
    fills = _point_fills(svg)
    assert len(fills) == 3
    mid_fill = _rgb(fills[1])  # the point at v=mid is the middle row
    assert _within(mid_fill, _rgb(stops[p_mid]), tol=1)


def test_diverging_scale_domain_mid_discriminates_across_the_same_domain():
    """Two charts sharing one domain but different domain_mid values must
    render differently -- an implementation that ignores domain_mid (treating
    the midpoint as the domain's geometric center regardless of input) would
    make these two identical."""
    a = _mid_chart(-1.0, 10.0, 3.0).to_svg()
    b = _mid_chart(-1.0, 10.0, 7.0).to_svg()
    assert a != b


def test_diverging_scale_domain_mid_is_not_symmetrized_across_domains():
    """Two charts with different, non-centered domains but the SAME p_mid
    offset must each carry the scheme's true mid color there (``scheme.sample(0.5)``,
    read from a no-midpoint render of the same scheme) -- not merely match
    each other, which a domain-symmetrizing implementation OR a
    p_mid-sampling implementation could also satisfy by being wrong in the
    same way on both. Proving each equals the true mid color also implies
    they equal each other, so this subsumes the weaker pairwise check."""
    svg_a = _mid_chart(-1.0, 10.0, 3.0).to_svg()  # center is 4.5, mid is 3
    svg_b = _mid_chart(-5.0, 6.0, -1.0).to_svg()  # center is 0.5, mid is -1
    p_mid = round((3.0 - -1.0) / (10.0 - -1.0), 4)
    stop_a = dict(_gradient_stops(svg_a))
    stop_b = dict(_gradient_stops(svg_b))
    true_mid = dict(_gradient_stops(_mid_chart(0.0, 1.0, None).to_svg()))[0.5]
    assert stop_a[p_mid] == true_mid
    assert stop_b[p_mid] == true_mid


def test_diverging_scale_three_element_domain_matches_domain_mid():
    """Brief case: '3-tuple domain=[lo, mid, hi] fixed at the one producer'.
    An off-center 3-element domain (not a geometric-center one, which would
    pass even if the 3-element route silently dropped its own midpoint)
    renders the identical bar to the equivalent domain=[lo, hi],
    domain_mid=mid chart, and carries the same p_mid stop."""
    lo, mid, hi = -1.0, 3.0, 10.0
    df = pl.DataFrame({"x": [1.0, 2.0, 3.0], "y": [1.0, 2.0, 3.0], "v": [lo, mid, hi]})
    three_element_svg = (
        fm.Chart(df)
        .mark_point()
        .encode(
            x="x:Q", y="y:Q", color=fm.Color("v", scale=fm.DivergingScale(domain=[lo, mid, hi]))
        )
        .properties(width=300, height=200)
        .to_svg()
    )
    domain_mid_svg = _mid_chart(lo, hi, mid).to_svg()
    assert three_element_svg == domain_mid_svg
    p_mid = round((mid - lo) / (hi - lo), 4)
    assert p_mid in dict(_gradient_stops(three_element_svg))


# ---------------------------------------------------------------------------
# midpoint=None / midpoint at the geometric center / midpoint outside domain
# ---------------------------------------------------------------------------


def test_midpoint_none_has_eleven_evenly_spaced_stops():
    svg = _mid_chart(-2.0, 2.0, None).to_svg()
    offsets = [o for o, _ in _gradient_stops(svg)]
    assert offsets == [round(i / 10, 4) for i in range(11)]


def test_midpoint_at_geometric_center_is_byte_identical_to_no_midpoint():
    none_svg = _mid_chart(-1.0, 11.0, None).to_svg()
    centered_svg = _mid_chart(-1.0, 11.0, 5.0).to_svg()  # geometric center of [-1, 11]
    assert none_svg == centered_svg


@pytest.mark.parametrize(
    ("lo", "hi"), [(-0.1, 0.5), (0.2, 0.8), (0.1, 0.7), (-0.3, 0.9), (-0.1, 0.7)]
)
def test_inexact_centered_domain_is_byte_identical_to_no_midpoint(lo, hi):
    """A geometric center computed as `(lo + hi) / 2` is rarely float-exact
    (e.g. `(-0.1 + 0.5) / 2 == 0.19999999999999998`); a midpoint at the
    geometric center still renders byte-identical to no midpoint.
    Proven within this build, not against a /tmp baseline: the
    inexact-centered chart must equal the sibling chart with no midpoint at
    all, over the same domain."""
    centered_svg = _mid_chart(lo, hi, (lo + hi) / 2).to_svg()
    no_midpoint_svg = _mid_chart(lo, hi, None).to_svg()
    assert centered_svg == no_midpoint_svg


def test_midpoint_outside_the_domain_has_eleven_stops_no_extra():
    df = pl.DataFrame({"x": [1.0, 2.0, 3.0], "y": [1.0, 2.0, 3.0], "v": [4.0, 7.0, 10.0]})
    chart = (
        fm.Chart(df)
        .mark_point()
        .encode(
            x="x:Q",
            y="y:Q",
            color=fm.Color("v", scale=fm.DivergingScale(domain=[4.0, 10.0], domain_mid=3.0)),
        )
        .properties(width=300, height=200)
    )
    svg = chart.to_svg()
    stops = _gradient_stops(svg)
    assert len(stops) == 11
    fills = _point_fills(svg)
    assert len(fills) == 3
    for value, fill in zip([4.0, 7.0, 10.0], fills, strict=True):
        bar = _bar_color_at(stops, 4.0, 10.0, value)
        assert _within(_rgb(fill), bar, tol=1), (value, fill, bar)


# ---------------------------------------------------------------------------
# heatmap(center=0.5) on [0.1, 0.9] with no cmap -> theme diverging scheme;
# explicit scheme wins; non-Diverging scales keep the sign heuristic.
# ---------------------------------------------------------------------------


def test_heatmap_center_with_positive_only_data_and_no_cmap_takes_the_theme_diverging_scheme():
    df = pl.DataFrame({"a": [0.1, 0.5, 0.9], "b": [0.2, 0.6, 0.8]})
    default_svg = fm.heatmap(df, annot=False, vmin=0.1, vmax=0.9, center=0.5).to_svg()
    theme_diverging_svg = fm.heatmap(
        df, annot=False, vmin=0.1, vmax=0.9, center=0.5, cmap="blue_to_red"
    ).to_svg()
    other_scheme_svg = fm.heatmap(
        df, annot=False, vmin=0.1, vmax=0.9, center=0.5, cmap="viridis"
    ).to_svg()
    assert default_svg == theme_diverging_svg, (
        "no cmap must default to the theme's diverging scheme"
    )
    assert default_svg != other_scheme_svg, "an explicit scheme must still win"


def test_non_diverging_scale_keeps_the_sign_heuristic():
    """Sanity control: a plain continuous (non-Diverging) color scale still
    picks sequential vs. diverging by the data's sign, unaffected by the
    Diverging-type scheme rule above. Compared by gradient stops, not whole
    SVG bytes -- two same-scheme charts over different data still differ in
    tick labels and point positions, so a whole-SVG inequality would pass
    even if the scheme never changed; a same-sign control (whose stops ARE
    equal) proves the stop difference below is the scheme, not that noise."""
    df = pl.DataFrame({"x": [1.0, 2.0, 3.0], "y": [1.0, 2.0, 3.0], "v": [1.0, 2.0, 3.0]})
    all_positive = fm.Chart(df).mark_point().encode(x="x:Q", y="y:Q", color="v:Q").to_svg()
    straddling_df = df.with_columns((pl.col("v") - 2).alias("v"))
    straddling = fm.Chart(straddling_df).mark_point().encode(x="x:Q", y="y:Q", color="v:Q").to_svg()
    assert _gradient_stops(all_positive) != _gradient_stops(straddling)

    shifted_df = df.with_columns((pl.col("v") + 10).alias("v"))
    shifted = fm.Chart(shifted_df).mark_point().encode(x="x:Q", y="y:Q", color="v:Q").to_svg()
    assert _gradient_stops(all_positive) == _gradient_stops(shifted)


# ---------------------------------------------------------------------------
# Tick labels stay linear in value
# ---------------------------------------------------------------------------


def test_tick_labels_stay_linear_in_value_with_a_midpoint():
    lo, hi, mid = -1.0, 10.0, 3.0
    svg = _mid_chart(lo, hi, mid).to_svg()
    ticks = [float(x) for x in _legend_tick_labels(svg)]
    assert len(ticks) == 5
    assert ticks[0] == pytest.approx(lo, abs=0.2)
    assert ticks[-1] == pytest.approx(hi, abs=0.2)
    diffs = [ticks[i + 1] - ticks[i] for i in range(4)]
    assert all(d == pytest.approx(diffs[0], abs=0.2) for d in diffs), ticks


# ---------------------------------------------------------------------------
# configure_color(range=)/(domain=)/(both): bar stops equal mark fills at
# stop values, static and interactive.
# ---------------------------------------------------------------------------


def _configured_chart(
    lo: float, hi: float, *, domain=None, range_=None, extra_values=()
) -> fm.Chart:
    """Five points at round offsets (0, 0.2, 0.5, 0.9, 1.0) of `[lo, hi]`, plus
    any `extra_values` appended as additional, unchecked rows.

    `extra_values` is how a `configure_color(domain=)` test proves the
    override actually took effect rather than coinciding with the data's own
    resolved domain by construction: a value outside `[lo, hi]` widens the
    chart's OWN pre-config domain past `[lo, hi]`, so passing
    `domain=[lo, hi]` to `configure_color` is a real override (shrinking the
    domain back down), not a no-op. If the override were silently ignored,
    the five checked points would be colored against the wider, un-configured
    domain instead, and `_assert_configured_stops_match_marks` would catch
    the mismatch.
    """
    offsets = [0.0, 0.2, 0.5, 0.9, 1.0]
    values = [lo + o * (hi - lo) for o in offsets] + list(extra_values)
    df = pl.DataFrame({"x": list(range(len(values))), "y": list(range(len(values))), "v": values})
    chart = (
        fm.Chart(df)
        .mark_point()
        .encode(x="x:Q", y="y:Q", color="v:Q")
        .properties(width=300, height=200)
    )
    return chart.configure_color(domain=domain, range=range_)


def _assert_configured_stops_match_marks(
    chart: fm.Chart, offsets: list[float], *, total_points: int
) -> None:
    svg = chart.to_svg()
    stops = dict(_gradient_stops(svg))
    fills = _point_fills(svg)
    assert len(fills) == total_points, (len(fills), total_points)
    for offset, fill in zip(offsets, fills[: len(offsets)], strict=True):
        key = round(offset, 4)
        assert key in stops, (key, stops)
        assert _within(_rgb(fill), _rgb(stops[key]), tol=1), (offset, fill, stops[key])


def test_configure_color_range_only_bar_matches_marks_static():
    chart = _configured_chart(-4.0, 4.0, range_=["red", "blue"])
    _assert_configured_stops_match_marks(chart, [0.0, 0.2, 0.5, 0.9, 1.0], total_points=5)


def test_configure_color_domain_only_relabels_the_bar_and_keeps_its_stops():
    """With no range override, a sequential/diverging bar's stops are a pure
    function of the scheme and the stop position -- domain-invariant whether
    or not a midpoint is set -- so `configure_color(domain=)` alone cannot
    change them. Pinned both ways: stops unchanged from
    the plain render, labels on the configured domain."""
    df = pl.DataFrame({"x": [1.0, 2.0, 3.0], "y": [1.0, 2.0, 3.0], "v": [1.0, 5.0, 9.0]})
    plain = (
        fm.Chart(df)
        .mark_point()
        .encode(x="x:Q", y="y:Q", color="v:Q")
        .properties(width=300, height=200)
    )
    configured = plain.configure_color(domain=[-4.0, 4.0])
    plain_svg = plain.to_svg()
    configured_svg = configured.to_svg()
    assert _gradient_stops(configured_svg) == _gradient_stops(plain_svg)
    ticks = [float(x) for x in _legend_tick_labels(configured_svg)]
    assert ticks[0] == pytest.approx(-4.0, abs=0.1)
    assert ticks[-1] == pytest.approx(4.0, abs=0.1)


def test_configure_color_domain_and_range_bar_matches_marks_static():
    # extra_values=[50.0] widens the chart's own pre-config domain to
    # [-4, 50], so configuring domain=[-4, 4] back down is a real override,
    # not a coincidence of the data's own extent (see _configured_chart).
    chart = _configured_chart(
        -4.0, 4.0, domain=[-4.0, 4.0], range_=["red", "blue"], extra_values=[50.0]
    )
    _assert_configured_stops_match_marks(chart, [0.0, 0.2, 0.5, 0.9, 1.0], total_points=6)
    _assert_bar_label_endpoints(chart.to_svg(), -4.0, 4.0)


def test_configure_color_with_midpoint_domain_matches_marks_static():
    """The "midpoint chart with configure_color(domain=)" case: a DivergingScale
    with its own domain_mid, reconfigured to a DIFFERENT domain after
    resolution -- the p_mid stop must move to the configured domain's own
    midpoint position and still match the marks' own color there.

    The configured domain [-2, 12] is deliberately unlike the data's own
    resolved domain [-1, 10] (which `DivergingScale(domain_mid=mid)` derives
    from the data when no explicit `domain=` is given on the scale): a
    configured domain equal to the data's own domain would be a no-op, and
    the test would pass even if `configure_color(domain=)` were silently
    ignored."""
    lo, mid, hi = -1.0, 3.0, 10.0
    configured_lo, configured_hi = -2.0, 12.0
    df = pl.DataFrame({"x": [1.0, 2.0, 3.0], "y": [1.0, 2.0, 3.0], "v": [lo, mid, hi]})
    chart = (
        fm.Chart(df)
        .mark_point()
        .encode(x="x:Q", y="y:Q", color=fm.Color("v", scale=fm.DivergingScale(domain_mid=mid)))
        .configure_color(domain=[configured_lo, configured_hi])
        .properties(width=300, height=200)
    )
    svg = chart.to_svg()
    stops = dict(_gradient_stops(svg))
    unconfigured_p_mid = round((mid - lo) / (hi - lo), 4)
    configured_p_mid = round((mid - configured_lo) / (configured_hi - configured_lo), 4)
    assert configured_p_mid in stops
    assert unconfigured_p_mid not in stops
    fills = _point_fills(svg)
    assert len(fills) == 3
    mid_fill = _rgb(fills[1])
    assert _within(mid_fill, _rgb(stops[configured_p_mid]), tol=1)


# ---------------------------------------------------------------------------
# configure_color: static and interactive parity
# ---------------------------------------------------------------------------


def _interactive_scene(chart: fm.Chart) -> dict:
    import json

    from ferrum._core import render_interactive

    spec, data, viewport, theme, chart_config = chart._render_inputs(_auto_tooltips=True)
    result = render_interactive(
        spec, data, viewport=viewport, theme=theme, chart_config=chart_config or None
    )
    json_str = result[0] if isinstance(result, tuple) else result
    return json.loads(json_str)


def _interactive_gradient_stops(scene: dict) -> dict[float, str]:
    def find_raw(nodes):
        out = []
        for n in nodes:
            if n.get("type") == "raw":
                out.append(n)
            elif n.get("type") == "group":
                out += find_raw(n.get("children", []))
        return out

    raw = []
    for region in ("title", "legend", "decorations"):
        raw += find_raw(scene.get(region, []))
    for panel in scene.get("panels", []):
        for region in ("grid", "below_marks", "axes", "chrome_above", "annotations", "strip_title"):
            raw += find_raw(panel.get(region, []))
    grad_nodes = [n for n in raw if "linearGradient" in n.get("svg", "")]
    assert len(grad_nodes) == 1, grad_nodes
    return dict(sorted((float(o), h) for o, h in _STOP_RE.findall(grad_nodes[0]["svg"])))


def _interactive_point_fills(scene: dict) -> list[tuple[int, int, int]]:
    nodes = scene["panels"][0]["marks"][0]["nodes"]
    return [
        (n["style"]["fill"]["r"], n["style"]["fill"]["g"], n["style"]["fill"]["b"]) for n in nodes
    ]


def _interactive_legend_tick_labels(scene: dict) -> list[str]:
    """The interactive scene's colorbar tick labels -- the `legend` region's
    `text` nodes anchored `"start"`, excluding the title (always first),
    mirroring `_legend_tick_labels`' static-SVG counterpart."""
    texts = [
        n["content"]
        for n in scene.get("legend", [])
        if n.get("type") == "text" and n.get("style", {}).get("anchor") == "start"
    ]
    return texts[1:]


def _assert_interactive_bar_label_endpoints(
    scene: dict, lo: float, hi: float, *, tol: float = 0.1
) -> None:
    ticks = [float(x) for x in _interactive_legend_tick_labels(scene)]
    assert ticks[0] == pytest.approx(lo, abs=tol), ticks
    assert ticks[-1] == pytest.approx(hi, abs=tol), ticks


def test_configure_color_domain_and_range_bar_matches_marks_interactive():
    # extra_values=[50.0] widens the chart's own pre-config domain, same as
    # the static counterpart, so the configured domain=[-4, 4] is a real
    # override and not coincident with the data's own extent.
    chart = _configured_chart(
        -4.0, 4.0, domain=[-4.0, 4.0], range_=["red", "blue"], extra_values=[50.0]
    )
    scene = _interactive_scene(chart)
    stops = _interactive_gradient_stops(scene)
    fills = _interactive_point_fills(scene)
    assert len(fills) == 6
    for offset, fill in zip([0.0, 0.2, 0.5, 0.9, 1.0], fills[:5], strict=True):
        key = round(offset, 4)
        assert key in stops
        assert _within(fill, _rgb(stops[key]), tol=1), (offset, fill, stops[key])
    _assert_interactive_bar_label_endpoints(scene, -4.0, 4.0)


def test_configure_color_with_midpoint_domain_matches_marks_interactive():
    # Configured domain [-2, 12] deliberately unlike the data's own resolved
    # domain [-1, 10] -- see the static counterpart's docstring.
    lo, mid, hi = -1.0, 3.0, 10.0
    configured_lo, configured_hi = -2.0, 12.0
    df = pl.DataFrame({"x": [1.0, 2.0, 3.0], "y": [1.0, 2.0, 3.0], "v": [lo, mid, hi]})
    chart = (
        fm.Chart(df)
        .mark_point()
        .encode(x="x:Q", y="y:Q", color=fm.Color("v", scale=fm.DivergingScale(domain_mid=mid)))
        .configure_color(domain=[configured_lo, configured_hi])
        .properties(width=300, height=200)
    )
    scene = _interactive_scene(chart)
    stops = _interactive_gradient_stops(scene)
    unconfigured_p_mid = round((mid - lo) / (hi - lo), 4)
    configured_p_mid = round((mid - configured_lo) / (configured_hi - configured_lo), 4)
    assert configured_p_mid in stops
    assert unconfigured_p_mid not in stops
    fills = _interactive_point_fills(scene)
    assert len(fills) == 3
    assert _within(fills[1], _rgb(stops[configured_p_mid]), tol=1)


# ---------------------------------------------------------------------------
# configure_color(range=) on a discretizing (quantize) scale
# ---------------------------------------------------------------------------


def test_configure_color_range_recolors_discretizing_swatches_and_matches_marks():
    df = pl.DataFrame({"x": [1.0, 2.0, 3.0], "y": [1.0, 2.0, 3.0], "v": [1.0, 4.0, 8.0]})
    chart = (
        fm.Chart(df)
        .mark_point()
        .encode(
            x="x:Q",
            y="y:Q",
            color=fm.Color(
                "v",
                scale=fm.QuantizeScale(domain=[0.0, 9.0], range=["#ff0000", "#00ff00", "#0000ff"]),
            ),
        )
        .configure_color(range=["#111111", "#222222", "#333333"])
        .properties(width=300, height=200)
    )
    svg = chart.to_svg()
    stops = _gradient_stops(svg)
    swatches = [hexstr for _, hexstr in stops[::2]]
    assert swatches == ["#111111", "#222222", "#333333"]
    fills = _point_fills(svg)
    assert fills == ["#111111", "#222222", "#333333"]


# ---------------------------------------------------------------------------
# configure_color(domain=): faceted shared bar, composite shared band
# ---------------------------------------------------------------------------


def test_configure_color_domain_faceted_shares_one_bar_matching_every_panel():
    # A decoy row (v=50.0, outside [-4, 4]) appended to facet "b" widens the
    # chart's own pre-config domain, so configuring domain=[-4, 4] is a real
    # override rather than coinciding with the data's own extent (see
    # _configured_chart's docstring for the same construction).
    df = pl.DataFrame(
        {
            "x": [1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 4.0],
            "y": [1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 4.0],
            "v": [-4.0, 0.0, 4.0, -4.0, 0.0, 4.0, 50.0],
            "g": ["a", "a", "a", "b", "b", "b", "b"],
        }
    )
    chart = (
        fm.Chart(df)
        .mark_point()
        .encode(x="x:Q", y="y:Q", color="v:Q")
        .facet(field="g")
        .configure_color(domain=[-4.0, 4.0], range=["red", "blue"])
        .properties(width=200, height=200)
    )
    svg = chart.to_svg()
    stop_blocks = svg.split("linearGradient")
    assert len(stop_blocks) == 3, "expected exactly one shared <linearGradient> (open+close split)"
    stops = dict(_gradient_stops(svg))
    fills = _point_fills(svg)
    assert len(fills) == 7
    for offset, fill in zip([0.0, 0.5, 1.0, 0.0, 0.5, 1.0], fills[:6], strict=True):
        key = round(offset, 4)
        assert _within(_rgb(fill), _rgb(stops[key]), tol=1), (offset, fill, stops[key])
    _assert_bar_label_endpoints(svg, -4.0, 4.0)


def test_configure_color_domain_on_composite_leaf_shared_band_reads_configured_scale():
    # A decoy row (val=100.0, outside [0, 20]) on leaf `a` widens its own
    # pre-config domain, so configuring domain=[0.0, 20.0] is a real override
    # rather than coinciding with leaf `a`'s own data extent (see
    # _configured_chart's docstring for the same construction).
    df1 = pl.DataFrame(
        {"x": [1.0, 2.0, 3.0, 4.0], "y": [1.0, 2.0, 3.0, 4.0], "val": [0.0, 10.0, 20.0, 100.0]}
    )
    df2 = pl.DataFrame({"x": [1.0, 2.0, 3.0], "y": [4.0, 5.0, 6.0], "val": [10.0, 2.0, 8.0]})
    a = (
        fm.Chart(df1)
        .mark_point()
        .encode(x="x", y="y", color="val")
        .configure_color(domain=[0.0, 20.0], range=["red", "blue"])
    )
    b = fm.Chart(df2).mark_point().encode(x="x", y="y", color="val")
    chart = fm.hconcat(a, b, resolve={"color": "shared"})
    svg = chart.to_svg()
    stop_blocks = svg.split("linearGradient")
    assert len(stop_blocks) == 3, "expected exactly one shared <linearGradient> (open+close split)"
    stops = dict(_gradient_stops(svg))
    fills = _point_fills(svg)
    assert len(fills) == 7
    # The first three circles are leaf `a`'s own points, at values 0, 10, 20 ->
    # offsets 0.0, 0.5, 1.0 of the configured [0, 20] domain -- the band this
    # fix makes read the configured scale. The decoy (index 3) and leaf `b`'s
    # own points (unconfigured, indices 4-6) are excluded.
    for offset, fill in zip([0.0, 0.5, 1.0], fills[:3], strict=True):
        key = round(offset, 4)
        assert _within(_rgb(fill), _rgb(stops[key]), tol=1), (offset, fill, stops[key])
    _assert_bar_label_endpoints(svg, 0.0, 20.0)
