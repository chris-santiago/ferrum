"""Net-new coverage for F-L09-04: catplot(hue_order=) silently ignored for
kind="box"/"violin".

``catplot`` builds ``Color(hue, sort=hue_order)`` correctly for every kind,
but ``_ResolveContext._color_field()`` flattened the color channel to a bare
field-name string before the composite desugars (``desugar_boxplot``,
``desugar_violin``, ``desugar_errorbar``, ``desugar_errorband``) ever saw it
-- so the box/violin hue-level order silently had no effect on rendered
output. ``color_sort`` now threads through ``_ResolveContext`` /
``_prep_color_field`` into those four desugars, which forward it to
``nominal_color_channel(field, sort=color_sort)``.

This module is scoped to the hue_order/color_sort regression per this repo's
Python test-file convention (see ``test_boxen_palette.py``'s module
docstring): a findings-ID-named module stays scoped to pinning the specific
regression the finding describes.
"""

from __future__ import annotations

import re

import numpy as np
import polars as pl

import ferrum as fm


def _legend_label_order(svg: str) -> list[str]:
    """Return the ``sex`` legend's label sequence in document order.

    The legend swatch (a ``<circle fill="...">``) immediately precedes its
    label (a ``<text>...</text>``) in ferrum's SVG output.
    """
    return re.findall(r'<circle[^>]*fill="[^"]+"/><text[^>]*>(Male|Female)<', svg)


def _violin_low_high_df() -> pl.DataFrame:
    # Violin needs more points per group than boxplot for a stable KDE body.
    return pl.DataFrame(
        {
            "cat": ["g"] * 40,
            "val": [1.0 + 0.05 * i for i in range(20)] + [8.0 + 0.05 * i for i in range(20)],
            "sex": (["Female"] * 20 + ["Male"] * 20),
        }
    )


def _tips_df() -> pl.DataFrame:
    # Mirrors the batch-F probe oracle's F-L09-04 fixture exactly (same seed,
    # same shape) so this test proves the regression against the real
    # catplot path, not a hand-built shape it never emits.
    rng = np.random.default_rng(0)
    return pl.DataFrame(
        {
            "day": rng.choice(["Thur", "Fri", "Sat", "Sun"], 200),
            "tip": rng.uniform(1, 10, 200),
            "sex": rng.choice(["Male", "Female"], 200),
        }
    )


def _boxplot_df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "day": ["Thur", "Fri", "Sat", "Sun"] * 10,
            "tip": [float(x % 7) + 1.0 for x in range(40)],
            "sex": (["Male", "Female"] * 20),
        }
    )


# ---------------------------------------------------------------------------
# catplot(hue_order=) — the probe's exact INERT pair (kind="box"/"violin")
# ---------------------------------------------------------------------------


def test_catplot_box_hue_order_swap_changes_output():
    df = _tips_df()
    svg_a = fm.catplot(
        df, x="day", y="tip", hue="sex", kind="box", hue_order=["Female", "Male"]
    ).to_svg()
    svg_b = fm.catplot(
        df, x="day", y="tip", hue="sex", kind="box", hue_order=["Male", "Female"]
    ).to_svg()
    assert svg_a != svg_b


def test_catplot_violin_hue_order_swap_changes_output():
    df = _tips_df()
    svg_a = fm.catplot(
        df, x="day", y="tip", hue="sex", kind="violin", hue_order=["Female", "Male"]
    ).to_svg()
    svg_b = fm.catplot(
        df, x="day", y="tip", hue="sex", kind="violin", hue_order=["Male", "Female"]
    ).to_svg()
    assert svg_a != svg_b


def test_catplot_box_hue_order_sets_legend_sequence():
    """Order-specific pin (quality review S2, cycle 1): a bare
    ``svg_a != svg_b`` only proves *some* difference, not that the
    difference is the correct reordering. This asserts the legend label
    sequence matches the *requested* ``hue_order`` exactly, for both
    spellings.

    A second assertion -- that the mark drawn over a data-identified
    low-value group carries the same-labeled legend's swatch fill -- was
    tried here and removed after cycle-2 quality review proved it holds in
    every state, including with the color-sort fix stripped out entirely
    (mutation-tested across 8 states: 2 kinds x 2 orders x baseline/mutated).
    One resolved color scale drives both the legend swatch and the mark
    fill, so that consistency is a renderer invariant a color-sort
    regression cannot break -- not a signal this fix needs pinned. All of
    this test's discriminating power lives in the assertion below.
    """
    df = _tips_df()
    for order in (["Female", "Male"], ["Male", "Female"]):
        svg = fm.catplot(df, x="day", y="tip", hue="sex", kind="box", hue_order=order).to_svg()
        assert _legend_label_order(svg)[:2] == order


def test_catplot_violin_hue_order_sets_legend_sequence():
    """Violin counterpart to ``test_catplot_box_hue_order_sets_legend_sequence``."""
    df = _tips_df()
    for order in (["Female", "Male"], ["Male", "Female"]):
        svg = fm.catplot(df, x="day", y="tip", hue="sex", kind="violin", hue_order=order).to_svg()
        assert _legend_label_order(svg)[:2] == order


def test_catplot_hue_order_none_byte_identical_for_box_and_violin():
    df = _tips_df()
    for kind in ("box", "violin"):
        svg_default = fm.catplot(df, x="day", y="tip", hue="sex", kind=kind).to_svg()
        svg_explicit_none = fm.catplot(
            df, x="day", y="tip", hue="sex", kind=kind, hue_order=None
        ).to_svg()
        assert svg_default == svg_explicit_none


# ---------------------------------------------------------------------------
# Bare mark_boxplot / mark_violin + .encode(color=Color(sort=...))
# ---------------------------------------------------------------------------


def test_bare_mark_boxplot_color_sort_honored():
    df = _boxplot_df()
    svg_a = (
        fm.Chart(df)
        .mark_boxplot()
        .encode(x="day", y="tip", color=fm.Color("sex", sort=["Female", "Male"]))
        .to_svg()
    )
    svg_b = (
        fm.Chart(df)
        .mark_boxplot()
        .encode(x="day", y="tip", color=fm.Color("sex", sort=["Male", "Female"]))
        .to_svg()
    )
    assert svg_a != svg_b


def test_bare_mark_violin_color_sort_honored():
    df = _boxplot_df()
    svg_a = (
        fm.Chart(df)
        .mark_violin()
        .encode(x="day", y="tip", color=fm.Color("sex", sort=["Female", "Male"]))
        .to_svg()
    )
    svg_b = (
        fm.Chart(df)
        .mark_violin()
        .encode(x="day", y="tip", color=fm.Color("sex", sort=["Male", "Female"]))
        .to_svg()
    )
    assert svg_a != svg_b


def test_bare_mark_violin_inner_point_color_sort_honored():
    """Coverage-gap pin (quality review S1, cycle 2): ``inner="point"``'s raw
    per-observation point layer is one of three ``nominal_color_channel``
    call sites in ``desugar_violin`` that had no dedicated test -- verified
    correct at review time, but unpinned. Cheap not-inert check."""
    df = _violin_low_high_df()
    svg_a = (
        fm.Chart(df)
        .mark_violin(inner="point")
        .encode(x="cat", y="val", color=fm.Color("sex", sort=["Female", "Male"]))
        .to_svg()
    )
    svg_b = (
        fm.Chart(df)
        .mark_violin(inner="point")
        .encode(x="cat", y="val", color=fm.Color("sex", sort=["Male", "Female"]))
        .to_svg()
    )
    assert svg_a != svg_b


def test_bare_mark_violin_inner_quartile_color_sort_honored():
    """Coverage-gap pin (quality review S1, cycle 2): ``inner="quartile"``'s
    per-tick loop is the second of three unpinned `desugar_violin`
    threading sites. Cheap not-inert check."""
    df = _violin_low_high_df()
    svg_a = (
        fm.Chart(df)
        .mark_violin(inner="quartile")
        .encode(x="cat", y="val", color=fm.Color("sex", sort=["Female", "Male"]))
        .to_svg()
    )
    svg_b = (
        fm.Chart(df)
        .mark_violin(inner="quartile")
        .encode(x="cat", y="val", color=fm.Color("sex", sort=["Male", "Female"]))
        .to_svg()
    )
    assert svg_a != svg_b


def test_bare_mark_boxplot_color_sort_none_byte_identical():
    df = _boxplot_df()
    svg_bare = fm.Chart(df).mark_boxplot().encode(x="day", y="tip", color="sex").to_svg()
    svg_explicit_none = (
        fm.Chart(df)
        .mark_boxplot()
        .encode(x="day", y="tip", color=fm.Color("sex", sort=None))
        .to_svg()
    )
    assert svg_bare == svg_explicit_none


# ---------------------------------------------------------------------------
# errorbar / errorband
# ---------------------------------------------------------------------------


def test_mark_errorbar_color_sort_honored():
    df = _boxplot_df()
    svg_a = (
        fm.Chart(df)
        .mark_errorbar()
        .encode(x="day", y="tip", color=fm.Color("sex", sort=["Female", "Male"]))
        .to_svg()
    )
    svg_b = (
        fm.Chart(df)
        .mark_errorbar()
        .encode(x="day", y="tip", color=fm.Color("sex", sort=["Male", "Female"]))
        .to_svg()
    )
    assert svg_a != svg_b


def test_mark_errorbar_color_sort_none_byte_identical():
    df = _boxplot_df()
    svg_bare = fm.Chart(df).mark_errorbar().encode(x="day", y="tip", color="sex").to_svg()
    svg_explicit_none = (
        fm.Chart(df)
        .mark_errorbar()
        .encode(x="day", y="tip", color=fm.Color("sex", sort=None))
        .to_svg()
    )
    assert svg_bare == svg_explicit_none


def test_mark_errorband_color_sort_honored():
    df = pl.DataFrame(
        {
            "x": list(range(20)) * 2,
            "tip": [float(x % 7) + 1.0 for x in range(40)],
            "sex": (["Male"] * 20 + ["Female"] * 20),
        }
    )
    svg_a = (
        fm.Chart(df)
        .mark_errorband()
        .encode(x="x", y="tip", color=fm.Color("sex", sort=["Female", "Male"]))
        .to_svg()
    )
    svg_b = (
        fm.Chart(df)
        .mark_errorband()
        .encode(x="x", y="tip", color=fm.Color("sex", sort=["Male", "Female"]))
        .to_svg()
    )
    assert svg_a != svg_b


def test_mark_errorband_borders_color_sort_honored():
    """Coverage-gap pin (quality review S1, cycle 2): ``borders=True`` adds
    the upper/lower border ``line`` layers, the third of three unpinned
    threading sites (`desugar_errorband`'s ``_enc_border`` closure). Cheap
    not-inert check."""
    df = pl.DataFrame(
        {
            "x": list(range(20)) * 2,
            "tip": [float(x % 7) + 1.0 for x in range(40)],
            "sex": (["Male"] * 20 + ["Female"] * 20),
        }
    )
    svg_a = (
        fm.Chart(df)
        .mark_errorband(borders=True)
        .encode(x="x", y="tip", color=fm.Color("sex", sort=["Female", "Male"]))
        .to_svg()
    )
    svg_b = (
        fm.Chart(df)
        .mark_errorband(borders=True)
        .encode(x="x", y="tip", color=fm.Color("sex", sort=["Male", "Female"]))
        .to_svg()
    )
    assert svg_a != svg_b


def test_mark_errorband_color_sort_none_byte_identical():
    df = pl.DataFrame(
        {
            "x": list(range(20)) * 2,
            "tip": [float(x % 7) + 1.0 for x in range(40)],
            "sex": (["Male"] * 20 + ["Female"] * 20),
        }
    )
    svg_bare = fm.Chart(df).mark_errorband().encode(x="x", y="tip", color="sex").to_svg()
    svg_explicit_none = (
        fm.Chart(df)
        .mark_errorband()
        .encode(x="x", y="tip", color=fm.Color("sex", sort=None))
        .to_svg()
    )
    assert svg_bare == svg_explicit_none


# ---------------------------------------------------------------------------
# boxen / swarm exclusions still hold — hue_order acts via the chart-level
# encoding cascade (the top-level `encoding.color` Rust reads for the
# shared legend), not via `_prep_color_field` / per-layer color threading,
# which these two kinds deliberately omit (documented load-bearing,
# _desugar.py). This pins that the exclusion still lets hue_order act, it
# just acts through a different mechanism than box/violin/errorbar/errorband.
# ---------------------------------------------------------------------------


def test_catplot_boxen_hue_order_swap_still_changes_output():
    df = _tips_df()
    svg_a = fm.catplot(
        df, x="day", y="tip", hue="sex", kind="boxen", hue_order=["Female", "Male"]
    ).to_svg()
    svg_b = fm.catplot(
        df, x="day", y="tip", hue="sex", kind="boxen", hue_order=["Male", "Female"]
    ).to_svg()
    assert svg_a != svg_b


def test_catplot_swarm_hue_order_swap_still_changes_output():
    df = _tips_df()
    svg_a = fm.catplot(
        df, x="day", y="tip", hue="sex", kind="swarm", hue_order=["Female", "Male"]
    ).to_svg()
    svg_b = fm.catplot(
        df, x="day", y="tip", hue="sex", kind="swarm", hue_order=["Male", "Female"]
    ).to_svg()
    assert svg_a != svg_b
