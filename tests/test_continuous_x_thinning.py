"""Continuous-x label thinning: the x-axis collision cascade's adjacent-gap
measure and its evenly-spaced stride thinning stage.

A crowded continuous x axis (linear, temporal or log) is judged by the gap
between ADJACENT labels, not every label against the axis's single widest
member, and before shrinking its font or rotating, it tries an evenly
spaced subset (an even stride, at least three survivors, flat at full
font) of its own ticks. Categorical and scale-less axes get the same
adjacent-gap measure but are never thinned. Every case here renders through
`fm.Chart(...).to_svg()` or a real composition of it -- no hand-built SVG
or layout fixture stands in for the renderer.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import numpy as np
import polars as pl
import pytest

import ferrum as fm
from ferrum._entry import EntryRefusal
from tests._script_loader import load_script

_REPO = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------------------
# Helpers -- a minimal SVG tick-label reader, scoped to this file. Mirrors
# the regex shape `tests/test_padding_auto_default.py` already uses for the
# same cascade's tests (text content, font-size, rotation, plot clip rect).
# ---------------------------------------------------------------------------

_TEXT = re.compile(r"<text([^>]*)>(.*?)</text>", re.S)
_ROTATE = re.compile(r"rotate\(([-\d.]+)")
_FONT_SIZE = re.compile(r'font-size="([\d.]+)"')
_CLIP_RECT = re.compile(r"<clipPath[^>]*><rect ([^>]*)/?>")
_ATTR = re.compile(r'([\w-]+)="([^"]*)"')


def _plot_rect(svg: str) -> dict[str, float]:
    """The first panel's plot clip rect: x, y, width, height."""
    attrs = dict(_ATTR.findall(_CLIP_RECT.findall(svg)[0]))
    return {k: float(v) for k, v in attrs.items()}


def _x_tick_labels(svg: str, rect: dict[str, float] | None = None) -> list[tuple[str, str, float]]:
    """Every x tick label below *rect*'s (or the first panel's) plot bottom,
    as ``(text, font_size, rotation_degrees)``, left to right by its own x
    position.

    A label's own line of text may be one `<text>` element (flat or
    rotated) or several (a wrapped multi-line label) -- each physical line
    is its own entry here, same as the renderer's own SVG shape.
    """
    rect = rect or _plot_rect(svg)
    bottom = rect["y"] + rect["height"]
    left_bound = rect["x"] - 15
    right_bound = rect["x"] + rect["width"] + 15
    entries = []
    for attrs, body in _TEXT.findall(svg):
        if "font-weight" in attrs:  # titles, never tick labels
            continue
        a = dict(_ATTR.findall(attrs))
        if "x" not in a or "y" not in a:
            continue
        y = float(a["y"])
        x = float(a["x"])
        if not (bottom < y < bottom + 60 and left_bound <= x <= right_bound):
            continue
        text = re.sub(r"<[^>]+>", "", body).strip()
        if not text:
            continue
        rot_m = _ROTATE.search(a.get("transform", ""))
        rotation = float(rot_m.group(1)) if rot_m else 0.0
        size_m = _FONT_SIZE.search(attrs)
        entries.append((x, text, size_m.group(1) if size_m else "", rotation))
    entries.sort(key=lambda e: e[0])
    return [(text, size, rot) for _, text, size, rot in entries]


def _facet_panels(svg: str, count: int = 2) -> list[dict[str, float]]:
    return [dict(_ATTR.findall(m)) for m in _CLIP_RECT.findall(svg)[:count]]


def _sha16(svg: str) -> str:
    return hashlib.sha256(svg.encode()).hexdigest()[:16]


def _scatter_df(seed: int, n: int = 120) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    return pl.DataFrame({"x": rng.normal(size=n), "y": rng.normal(size=n)})


# ---------------------------------------------------------------------------
# Sparse continuous x: every tick kept, flat, at full font -- the
# adjacent-pair measure alone already fits them, no thinning needed.
# ---------------------------------------------------------------------------


def test_sparse_continuous_x_keeps_every_label_flat_at_full_font():
    """A continuous x axis with room between adjacent labels draws every
    one of its own ticks flat at the full 11pt font -- `cull_labels=False`
    (which only disables thinning) renders byte-identically, proving
    nothing was thinned away."""
    df = _scatter_df(seed=7)
    chart = fm.Chart(df).mark_point().encode(x="x", y="y").properties(width=400, height=300)
    svg_default = chart.to_svg()
    svg_no_thin = chart.theme(fm.Theme(cull_labels=False)).to_svg()
    assert svg_default == svg_no_thin
    labels = _x_tick_labels(svg_default)
    assert len(labels) == 10
    assert all(size == "11" and rot == 0.0 for _, size, rot in labels)
    assert [t for t, _, _ in labels] == [
        "-2.5",
        "-2",
        "-1.5",
        "-1",
        "-0.5",
        "0",
        "0.5",
        "1",
        "1.5",
        "2",
    ]


# ---------------------------------------------------------------------------
# Crowded continuous x: thinned to an even stride, flat, at full font. The
# discriminator: the adjacent-gap measure alone (`cull_labels=False`, which
# disables only thinning) leaves the same axis rotated at -90 -- thinning is
# what rescues it. `cull_threshold=0` still keeps every label flat: the
# pair test floors at the slot-width tolerance, not a bare pixel gap of
# zero. The stride counts from this axis's own zero tick, not index 0, so
# the survivors include 0 rather than landing on an all-half-step set.
# ---------------------------------------------------------------------------


def test_crowded_continuous_x_thins_to_an_even_stride():
    df = _scatter_df(seed=7)
    chart = fm.Chart(df).mark_point().encode(x="x", y="y").properties(width=300, height=300)

    svg_default = chart.to_svg()
    labels = _x_tick_labels(svg_default)
    assert len(labels) == 5
    assert all(size == "11" and rot == 0.0 for _, size, rot in labels)
    assert [t for t, _, _ in labels] == ["-2", "-1", "0", "1", "2"]

    # Discriminator: the adjacent-gap measure alone (no thinning) leaves
    # this same crowded axis rotated at -90, not flat.
    svg_no_thin = chart.theme(fm.Theme(cull_labels=False)).to_svg()
    no_thin_labels = _x_tick_labels(svg_no_thin)
    assert len(no_thin_labels) == 10
    assert all(rot == -90.0 for _, _, rot in no_thin_labels)

    # `cull_threshold=0` disables the floor on the GAP term only; the
    # tolerance term (a fraction of the slot width) still applies, so this
    # axis stays flat and unthinned rather than rotating.
    svg_zero_threshold = chart.theme(fm.Theme(cull_threshold=0)).to_svg()
    zero_threshold_labels = _x_tick_labels(svg_zero_threshold)
    assert len(zero_threshold_labels) == 10
    assert all(size == "11" and rot == 0.0 for _, size, rot in zero_threshold_labels)


def test_thinned_and_unthinned_flat_bands_reserve_the_same_height():
    """The band-budget estimate that reserves vertical room for the x axis
    uses the same measure the drawn cascade does. At 220px this axis
    thins to 5 labels flat at full font; at 400px it already draws every
    label flat without thinning. Both reserve the same, short, single-line
    band height (224.961px of plot), proving the reservation tracks the
    drawn presentation rather than a wider, stale estimate: with thinning
    disabled (`cull_labels=False`), the same 220px chart instead rotates
    its labels and reserves a much taller band (205.577px of plot)."""
    df = _scatter_df(seed=7)
    narrow = fm.Chart(df).mark_point().encode(x="x", y="y").properties(width=220, height=300)
    wide = fm.Chart(df).mark_point().encode(x="x", y="y").properties(width=400, height=300).to_svg()
    narrow_svg = narrow.to_svg()
    narrow_labels = _x_tick_labels(narrow_svg)
    assert [t for t, _, _ in narrow_labels] == ["-2", "-1", "0", "1", "2"]
    assert all(size == "11" and rot == 0.0 for _, size, rot in narrow_labels)
    narrow_height = _plot_rect(narrow_svg)["height"]
    wide_height = _plot_rect(wide)["height"]
    assert narrow_height == wide_height == 224.961

    rotated_svg = narrow.theme(fm.Theme(cull_labels=False)).to_svg()
    rotated_labels = _x_tick_labels(rotated_svg)
    assert all(rot == -90.0 for _, _, rot in rotated_labels)
    assert _plot_rect(rotated_svg)["height"] == 205.577


# ---------------------------------------------------------------------------
# Independent per-panel thinning: two panels over different domains thin
# (or don't) on their own, not on one shared decision.
# ---------------------------------------------------------------------------


def test_facet_panels_thin_independently():
    rng = np.random.default_rng(7)
    wide = pl.DataFrame({"x": rng.normal(size=120), "y": rng.normal(size=120), "g": "a"})
    narrow = pl.DataFrame(
        {"x": rng.normal(loc=0.0, scale=0.15, size=120), "y": rng.normal(size=120), "g": "b"}
    )
    df = pl.concat([wide, narrow])
    chart = (
        fm.Chart(df)
        .mark_point()
        .encode(x="x", y="y")
        .properties(width=320, height=250)
        .facet(col="g")
        .share_scale(x="independent")
    )
    svg = chart.to_svg()
    panels = _facet_panels(svg)
    rect0 = {k: float(v) for k, v in panels[0].items()}
    rect1 = {k: float(v) for k, v in panels[1].items()}
    labels0 = _x_tick_labels(svg, rect0)
    labels1 = _x_tick_labels(svg, rect1)
    assert [t for t, _, _ in labels0] == ["-2", "-1", "0", "1", "2"]
    assert [t for t, _, _ in labels1] == ["-0.3", "0", "0.3"]
    assert all(size == "11" and rot == 0.0 for _, size, rot in labels0 + labels1)


def test_narrow_facet_panel_thins_a_shared_continuous_axis():
    rng = np.random.default_rng(0)
    df = pl.DataFrame({"x": rng.normal(size=200), "y": rng.normal(size=200), "g": ["a", "b"] * 100})
    chart = (
        fm.Chart(df)
        .mark_point()
        .encode(x="x", y="y")
        .properties(width=240, height=200)
        .facet(col="g")
    )
    svg = chart.to_svg()
    panels = _facet_panels(svg)
    for panel in panels:
        rect = {k: float(v) for k, v in panel.items()}
        assert round(rect["width"], 1) == 72.3
        labels = _x_tick_labels(svg, rect)
        assert [t for t, _, _ in labels] == ["-1.5", "0", "1.5"]
        assert all(size == "11" and rot == 0.0 for _, size, rot in labels)


# ---------------------------------------------------------------------------
# Categorical and scale-less axes: a crowded set wraps or shrinks under the
# adjacent-gap measure, but they are never thinned -- stride culling is a
# continuous-axis-only stage.
# ---------------------------------------------------------------------------


def test_categorical_axis_gets_the_measure_but_never_thins():
    cats = pl.DataFrame({"c": [f"Region {i}" for i in range(9)], "v": list(range(9))})

    wide = fm.Chart(cats).mark_bar().encode(x="c", y="v").properties(width=480, height=300).to_svg()
    wide_labels = _x_tick_labels(wide)
    # Every one of the 9 categories still draws -- wrapped onto two lines
    # (category name, number), not reduced to a stride subset.
    assert len(wide_labels) == 18
    assert all(size == "11" and rot == 0.0 for _, size, rot in wide_labels)

    narrow = (
        fm.Chart(cats).mark_bar().encode(x="c", y="v").properties(width=360, height=300).to_svg()
    )
    narrow_labels = _x_tick_labels(narrow)
    # Still every category -- rotated rather than thinned away.
    assert len(narrow_labels) == 9
    assert all(rot == -90.0 for _, _, rot in narrow_labels)
    assert [t for t, _, _ in narrow_labels] == [f"Region {i}" for i in range(9)]


# ---------------------------------------------------------------------------
# Temporal and log x: the same stride-thinning stage, reusing
# `anchored_visibility` for the decade anchors on a log axis.
# ---------------------------------------------------------------------------


def test_temporal_x_thins_like_continuous_linear_x():
    import datetime as dt

    dates = [dt.date(2024, 1, 1) + dt.timedelta(days=i) for i in range(0, 60, 3)]
    df = pl.DataFrame({"d": dates, "v": list(range(len(dates)))})
    chart = fm.Chart(df).mark_line().encode(x="d:T", y="v:Q").properties(width=300, height=250)

    svg_default = chart.to_svg()
    labels = _x_tick_labels(svg_default)
    assert [t for t, _, _ in labels] == ["2024-01-02", "2024-01-22", "2024-02-11"]
    assert all(size == "11" and rot == 0.0 for _, size, rot in labels)

    # Discriminator: the measure alone (no thinning) leaves the same dense
    # date axis rotated, not flat.
    svg_no_thin = chart.theme(fm.Theme(cull_labels=False)).to_svg()
    no_thin_labels = _x_tick_labels(svg_no_thin)
    assert len(no_thin_labels) == 29
    assert all(rot == -90.0 for _, _, rot in no_thin_labels)


def test_log_x_thins_via_decade_anchors():
    x = np.array([10.0**i for i in np.linspace(0, 6, 40)])
    df = pl.DataFrame({"x": x, "y": list(range(len(x)))})
    chart = (
        fm.Chart(df)
        .mark_point()
        .encode(x=fm.X("x", scale=fm.LogScale()), y="y")
        .properties(width=260, height=250)
    )

    svg_default = chart.to_svg()
    labels = _x_tick_labels(svg_default)
    # Every other decade survives -- the stride walks the log scale's own
    # decade anchors, not a plain index-modulo cull.
    assert [t for t, _, _ in labels] == ["1", "100", "10000", "1e6"]
    assert all(size == "11" and rot == 0.0 for _, size, rot in labels)

    svg_no_thin = chart.theme(fm.Theme(cull_labels=False)).to_svg()
    no_thin_labels = _x_tick_labels(svg_no_thin)
    assert [t for t, _, _ in no_thin_labels] == ["1", "10", "100", "1000", "10000", "100000", "1e6"]
    assert all(rot == -90.0 for _, _, rot in no_thin_labels)


# ---------------------------------------------------------------------------
# No stride with at least three survivors: the cascade falls through to
# shrinking, rotation, culling and elision, in that order.
# ---------------------------------------------------------------------------


def test_no_qualifying_stride_falls_through_to_rotation():
    """Four candidate ticks only: a stride of 2 leaves 2 survivors, under
    the 3-label floor, so no stride ever qualifies and the cascade rotates
    instead."""
    df = pl.DataFrame({"x": [0.0, 1000.0, 2000.0, 3000.0], "y": [0.0, 1.0, 2.0, 3.0]})
    chart = (
        fm.Chart(df)
        .mark_point()
        .encode(
            x=fm.X("x", axis=fm.Axis(label_format=",.6f", values=[0.0, 1000.0, 2000.0, 3000.0])),
            y="y",
        )
        .properties(width=150, height=250)
    )
    svg = chart.to_svg()
    labels = _x_tick_labels(svg)
    assert len(labels) == 4
    assert all(rot == -90.0 for _, _, rot in labels)
    assert [t for t, _, _ in labels] == [
        "0.000000",
        "1,000.000000",
        "2,000.000000",
        "3,000.000000",
    ]


# ---------------------------------------------------------------------------
# Byte-identical controls: an explicit `label_angle`, an explicit
# `label_overlap` spelling and every y axis are untouched by the thinning
# stage -- pinned against the stage baseline's own bytes.
# ---------------------------------------------------------------------------


def _controls_chart():
    df = _scatter_df(seed=0, n=200)
    return fm.Chart(df).mark_point().encode(x="x", y="y").properties(width=480, height=300)


@pytest.mark.parametrize(
    "axis_kwargs, expected_hash",
    [
        ({"label_angle": -45}, "5e6a3b680f2c2300"),
        ({"label_overlap": "parity"}, "45415e25af23e88f"),
        ({"label_overlap": "true"}, "2123825f51865424"),
        ({"label_overlap": "rotate"}, "ae3703706faecb7d"),
    ],
)
def test_explicit_angle_and_overlap_controls_are_byte_identical(axis_kwargs, expected_hash):
    chart = _controls_chart().encode(x=fm.X("x", axis=fm.Axis(**axis_kwargs)), y="y")
    assert _sha16(chart.to_svg()) == expected_hash


def test_explicit_angle_control_keeps_every_label_at_its_angle():
    chart = _controls_chart().encode(x=fm.X("x", axis=fm.Axis(label_angle=-45)), y="y")
    labels = _x_tick_labels(chart.to_svg())
    assert len(labels) == 9
    assert all(rot == -45.0 for _, _, rot in labels)


def test_label_overlap_parity_thins_by_its_own_rule_not_the_cascades():
    chart = _controls_chart().encode(x=fm.X("x", axis=fm.Axis(label_overlap="parity")), y="y")
    labels = _x_tick_labels(chart.to_svg())
    assert len(labels) == 5
    assert all(size == "11" and rot == 0.0 for _, size, rot in labels)


def test_label_overlap_parity_counts_from_index_zero_not_the_axis_origin():
    """`label_overlap="parity"` keeps every other label by plain index
    parity, not the thinning stage's own axis-origin phase. On this axis
    the zero tick sits at index 5 (odd): index-0 parity keeps the even
    indices and drops zero ("-2.5", "-1.5", "-0.5", "0.5", "1.5", "2.5");
    origin-phased parity would instead keep the odd indices and include it
    ("-2", "-1", "0", "1", "2", "3"). The drawn set is the index-0 one."""
    rng = np.random.default_rng(5)
    df = pl.DataFrame({"x": rng.normal(loc=0.0, scale=1.3, size=150), "y": rng.normal(size=150)})
    chart = (
        fm.Chart(df)
        .mark_point()
        .encode(x=fm.X("x", axis=fm.Axis(label_overlap="parity")), y="y")
        .properties(width=480, height=300)
    )
    labels = _x_tick_labels(chart.to_svg())
    assert [t for t, _, _ in labels] == ["-2.5", "-1.5", "-0.5", "0.5", "1.5", "2.5"]


def test_a_domain_starting_at_zero_is_byte_identical():
    """An axis whose own origin already sits at index 0 (a domain from 0
    upward) is unaffected by origin-phased thinning -- byte-identical to
    this pinned hash."""
    df = pl.DataFrame({"x": [0.0, 1_000_000.0], "y": [0.0, 1.0]})
    chart = (
        fm.Chart(df)
        .mark_point()
        .encode(x=fm.X("x:Q", axis=fm.Axis(label_format="currency")), y="y:Q")
        .properties(width=480, height=300)
    )
    assert _sha16(chart.to_svg()) == "b4826c11be4f4a7d"


def test_every_y_axis_is_unaffected_by_x_thinning():
    """A y-heavy chart (a dense continuous y axis, a categorical x so the x
    side never enters the cascade at all) renders byte-identically to the
    pinned hash: the y axis's own collision handling never calls the
    adjacent-gap measure or the thinning stage."""
    rng = np.random.default_rng(3)
    df = pl.DataFrame({"cat": rng.choice(["a", "b", "c"], size=200), "val": rng.normal(size=200)})
    chart = fm.Chart(df).mark_point().encode(x="cat:N", y="val:Q").properties(width=300, height=400)
    assert _sha16(chart.to_svg()) == "e8f93a68657b08f7"


# ---------------------------------------------------------------------------
# Gallery cards with a sparse or crowded continuous x axis, pinned against
# their own real chart-building code. `_scatter_theme_chart` is the one
# exception: it mirrors `design-docs/themes/demo_themes.py`'s `build_scatter`
# rather than calling it directly, because that builder returns PNG bytes
# instead of a `Chart`.
# ---------------------------------------------------------------------------

_GRAMMAR_MARKS = load_script(
    "demo_grammar_marks_thinning_pin", _REPO / "design-docs/demos/demo_grammar_marks.py"
)
_THEMES_DEMO = load_script("demo_themes_thinning_pin", _REPO / "design-docs/themes/demo_themes.py")
_SHOWCASE = load_script(
    "generate_showcase_pngs_thinning_pin", _REPO / "scripts/generate-showcase-pngs.py"
)
_DIAGNOSTICS_DEMO = load_script(
    "demo_diagnostics_thinning_pin", _REPO / "design-docs/demos/demo_diagnostics.py"
)


@pytest.mark.parametrize(
    "builder, expected_count",
    [
        (_GRAMMAR_MARKS.build_mark_point, 13),
        (_GRAMMAR_MARKS.build_mark_hex, 14),
        (_GRAMMAR_MARKS.build_mark_segment, 18),
        (_GRAMMAR_MARKS.build_mark_tick, 14),
        (_GRAMMAR_MARKS.build_mark_density, 14),
    ],
)
def test_sparse_gallery_mark_cards_draw_every_label_flat(builder, expected_count):
    labels = _x_tick_labels(builder().to_svg())
    assert len(labels) == expected_count
    assert all(size == "11" and rot == 0.0 for _, size, rot in labels)


def _scatter_theme_chart(theme) -> fm.Chart:
    """Mirrors `demo_themes.py`'s `build_scatter`, field for field, since
    that function returns PNG bytes and this needs the `Chart` itself to
    call `.to_svg()` on."""
    return (
        fm.Chart(_THEMES_DEMO.df_scatter)
        .mark_point()
        .encode(
            x=fm.X("x", title="Component 1"),
            y=fm.Y("y", title="Component 2"),
            color=fm.Color("class", title="Class"),
        )
        .properties(title="Cluster Scatter", width=480, height=300)
        .theme(theme)
    )


@pytest.mark.parametrize("theme_name", ["paper_ink", "arctic_signal", "slate_citrus"])
def test_scatter_theme_cards_thin_to_seven_labels(theme_name):
    theme = getattr(fm.themes, theme_name)
    svg = _scatter_theme_chart(theme).to_svg()
    labels = _x_tick_labels(svg)
    assert [t for t, _, _ in labels] == ["-3", "-2", "-1", "0", "1", "2", "3"]
    assert all(size == "11" and rot == 0.0 for _, size, rot in labels)
    assert round(_plot_rect(svg)["height"], 1) == 203.2


def test_gapminder_bubble_card_thins_to_five_labels():
    svg = _SHOWCASE.build_gapminder_bubble().to_svg()
    labels = _x_tick_labels(svg)
    assert len(labels) == 5
    assert all(size == "11" and rot == 0.0 for _, size, rot in labels)
    assert round(_plot_rect(svg)["height"], 1) == 249.9


def test_pdp_chart_card_thins_each_panel_independently():
    svg = _DIAGNOSTICS_DEMO.build_pdp_chart().to_svg()
    panels = _facet_panels(svg)
    f0, f1 = ({k: float(v) for k, v in panel.items()} for panel in panels)
    f0_labels = _x_tick_labels(svg, f0)
    f1_labels = _x_tick_labels(svg, f1)
    assert [t for t, _, _ in f0_labels] == ["4.4", "5", "5.6", "6.2", "6.8", "7.4"]
    assert [t for t, _, _ in f1_labels] == ["2", "2.4", "2.8", "3.2", "3.6", "4", "4.4"]
    assert all(size == "11" and rot == 0.0 for _, size, rot in f0_labels + f1_labels)


# ---------------------------------------------------------------------------
# NaN/inf viewports terminate.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("width", [float("nan"), float("inf"), float("-inf")])
def test_nan_and_inf_viewports_are_refused(width):
    df = pl.DataFrame({"x": [1.0, 2.0, 3.0], "y": [1.0, 2.0, 3.0]})
    chart = fm.Chart(df).mark_point().encode(x="x", y="y")
    with pytest.raises(EntryRefusal, match="finite"):
        chart.properties(width=width, height=200)
